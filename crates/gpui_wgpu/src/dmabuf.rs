//! Importing a frame another process shares as a Linux DMA-BUF (Chromium's off-screen
//! rendering with shared textures) and copying it into an atlas tile on the GPU, so its
//! pixels never pass through the CPU.

use anyhow::{Context as _, Result, anyhow, bail};
use ash::vk;
use gpui::{Bounds, DevicePixels, DmaBufFrame};
use std::os::fd::{FromRawFd as _, IntoRawFd as _, OwnedFd};
use wgpu::hal::api::Vulkan;

const fn fourcc(code: &[u8; 4]) -> u32 {
    u32::from_le_bytes(*code)
}

/// DRM_FORMAT_MOD_INVALID: the producer did not say how the buffer is laid out.
const MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

fn formats(code: u32) -> Option<(vk::Format, wgpu::TextureFormat)> {
    // DRM fourccs name channels from the most significant bit of a little-endian word, so
    // ARGB8888 is B, G, R, A in memory.
    if code == fourcc(b"AR24") || code == fourcc(b"XR24") {
        Some((vk::Format::B8G8R8A8_UNORM, wgpu::TextureFormat::Bgra8Unorm))
    } else if code == fourcc(b"AB24") || code == fourcc(b"XB24") {
        Some((vk::Format::R8G8B8A8_UNORM, wgpu::TextureFormat::Rgba8Unorm))
    } else {
        None
    }
}

/// Copies `frame` into `bounds` of `target` and waits until the GPU has done it.
pub(crate) fn copy_into(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frame: &DmaBufFrame,
    target: &wgpu::Texture,
    bounds: Bounds<DevicePixels>,
) -> Result<()> {
    let (vk_format, format) =
        formats(frame.fourcc).with_context(|| format!("unsupported fourcc {:#x}", frame.fourcc))?;
    if format != target.format() {
        bail!("frame is {format:?}, the atlas {:?}", target.format());
    }
    // The buffer may be padded past what is shown (its coded size); the tile is the shown
    // size, which starts at the buffer's corner.
    let width = frame.width.min(bounds.size.width.0 as u32);
    let height = frame.height.min(bounds.size.height.0 as u32);
    if frame.planes.is_empty() {
        bail!("frame has no planes");
    }
    if frame.modifier == MOD_INVALID {
        bail!("frame has no format modifier");
    }

    let imported = unsafe { import(device, frame, vk_format, format)? };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("dmabuf copy"),
    });
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &imported,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: target,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: bounds.origin.x.0 as u32,
                y: bounds.origin.y.0 as u32,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let submission = queue.submit([encoder.finish()]);
    // The producer may reuse the buffer as soon as we return.
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .map_err(|e| anyhow!("waiting for the copy: {e}"))?;
    drop(imported);
    Ok(())
}

/// Wraps the DMA-BUF in a Vulkan image wgpu can copy from. The image owns a dup of the
/// fd, so the producer's stays its own.
unsafe fn import(
    device: &wgpu::Device,
    frame: &DmaBufFrame,
    vk_format: vk::Format,
    format: wgpu::TextureFormat,
) -> Result<wgpu::Texture> {
    let hal_device = unsafe { device.as_hal::<Vulkan>() }.context("not a Vulkan device")?;
    let extensions = hal_device.enabled_device_extensions();
    for needed in [
        ash::khr::external_memory_fd::NAME,
        ash::ext::external_memory_dma_buf::NAME,
        ash::ext::image_drm_format_modifier::NAME,
    ] {
        if !extensions.contains(&needed) {
            bail!("device lacks {needed:?}");
        }
    }
    let raw = hal_device.raw_device();
    let instance = hal_device.shared_instance().raw_instance();
    let physical = hal_device.raw_physical_device();

    let layouts: Vec<vk::SubresourceLayout> = frame
        .planes
        .iter()
        .map(|p| vk::SubresourceLayout {
            offset: p.offset,
            size: 0,
            row_pitch: p.stride as u64,
            array_pitch: 0,
            depth_pitch: 0,
        })
        .collect();
    let mut modifier_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(frame.modifier)
        .plane_layouts(&layouts);
    let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk_format)
        .extent(vk::Extent3D {
            width: frame.width,
            height: frame.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external_info)
        .push_next(&mut modifier_info);
    let image = unsafe { raw.create_image(&image_info, None) }.context("vkCreateImage")?;

    let memory = match unsafe { bind_memory(raw, instance, physical, image, frame) } {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { raw.destroy_image(image, None) };
            return Err(error);
        }
    };

    let desc = wgpu::TextureDescriptor {
        label: Some("dmabuf"),
        size: wgpu::Extent3d {
            width: frame.width,
            height: frame.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };
    let hal_desc = wgpu::hal::TextureDescriptor {
        label: Some("dmabuf"),
        size: desc.size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUses::COPY_SRC,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };
    // wgpu destroys the image and frees the memory when the texture drops.
    let hal_texture = unsafe {
        hal_device.texture_from_raw(
            image,
            &hal_desc,
            None,
            wgpu::hal::vulkan::TextureMemory::Dedicated(memory),
        )
    };
    drop(hal_device);
    Ok(unsafe { device.create_texture_from_hal::<Vulkan>(hal_texture, &desc) })
}

unsafe fn bind_memory(
    raw: &ash::Device,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    image: vk::Image,
    frame: &DmaBufFrame,
) -> Result<vk::DeviceMemory> {
    // Vulkan takes the fd it imports; hand it a dup so the producer's stays open.
    let fd = unsafe { libc_dup(frame.planes[0].fd) }?;

    let fd_api = ash::khr::external_memory_fd::Device::new(instance, raw);
    let mut fd_props = vk::MemoryFdPropertiesKHR::default();
    unsafe {
        fd_api.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            std::os::fd::AsRawFd::as_raw_fd(&fd),
            &mut fd_props,
        )
    }
    .context("vkGetMemoryFdPropertiesKHR")?;

    let requirements = unsafe { raw.get_image_memory_requirements(image) };
    let properties = unsafe { instance.get_physical_device_memory_properties(physical) };
    let bits = requirements.memory_type_bits & fd_props.memory_type_bits;
    let type_index = (0..properties.memory_type_count)
        .find(|i| bits & (1 << i) != 0)
        .context("no memory type fits the imported buffer")?;

    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let raw_fd = fd.into_raw_fd();
    let mut import = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(raw_fd);
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(type_index)
        .push_next(&mut import)
        .push_next(&mut dedicated);
    let memory = match unsafe { raw.allocate_memory(&alloc, None) } {
        Ok(memory) => memory,
        Err(error) => {
            // On failure the fd is still ours to close.
            drop(unsafe { OwnedFd::from_raw_fd(raw_fd) });
            return Err(anyhow!("vkAllocateMemory: {error}"));
        }
    };
    if let Err(error) = unsafe { raw.bind_image_memory(image, memory, 0) } {
        unsafe { raw.free_memory(memory, None) };
        bail!("vkBindImageMemory: {error}");
    }
    Ok(memory)
}

unsafe fn libc_dup(fd: i32) -> Result<OwnedFd> {
    let borrowed = unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) };
    borrowed
        .try_clone_to_owned()
        .context("dup of the DMA-BUF fd")
}
