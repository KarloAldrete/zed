use anyhow::{Context as _, Result};
use etagere::{BucketedAtlasAllocator, size2};
use gpui::{
    AtlasBackend, AtlasKey, AtlasState, AtlasTextureId, AtlasTextureKind, AtlasTextureList,
    AtlasTile, Bounds, DevicePixels, PlatformAtlas, Point, Size,
};
use parking_lot::Mutex;
use std::{borrow::Cow, ops, sync::Arc};

use crate::WgpuContext;

fn device_size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    size2(size.width.0, size.height.0)
}

fn etagere_point_to_device(point: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels(point.x),
        y: DevicePixels(point.y),
    }
}

pub struct WgpuAtlas(Mutex<AtlasState<WgpuAtlasTextures>>);

struct PendingUpload {
    id: AtlasTextureId,
    bounds: Bounds<DevicePixels>,
    data: Vec<u8>,
}

struct WgpuAtlasTextures {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    max_texture_size: u32,
    color_texture_format: wgpu::TextureFormat,
    storage: WgpuAtlasStorage,
    pending_uploads: Vec<PendingUpload>,
}

pub struct WgpuTextureInfo {
    pub view: wgpu::TextureView,
}

impl WgpuAtlas {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        color_texture_format: wgpu::TextureFormat,
    ) -> Self {
        let max_texture_size = device.limits().max_texture_dimension_2d;
        WgpuAtlas(Mutex::new(AtlasState::new(WgpuAtlasTextures {
            device,
            queue,
            max_texture_size,
            color_texture_format,
            storage: WgpuAtlasStorage::default(),
            pending_uploads: Vec::new(),
        })))
    }

    pub fn from_context(context: &WgpuContext) -> Self {
        Self::new(
            context.device.clone(),
            context.queue.clone(),
            context.color_texture_format(),
        )
    }

    pub fn before_frame(&self) {
        let mut lock = self.0.lock();
        lock.backend.flush_uploads();
    }

    pub fn get_texture_info(&self, id: AtlasTextureId) -> WgpuTextureInfo {
        let lock = self.0.lock();
        let texture = &lock.backend.storage[id];
        WgpuTextureInfo {
            view: texture.view.clone(),
        }
    }

    /// Clears all cached textures and tiles, forcing them to be recreated.
    /// Use this for incremental recovery when the device is still valid.
    pub fn clear(&self) {
        self.0.lock().clear(|textures| {
            textures.storage = WgpuAtlasStorage::default();
            textures.pending_uploads.clear();
        });
    }

    /// Handles device lost by clearing all textures and cached tiles.
    /// The atlas will lazily recreate textures as needed on subsequent frames.
    pub fn handle_device_lost(&self, context: &WgpuContext) {
        self.0.lock().clear(|textures| {
            textures.device = context.device.clone();
            textures.queue = context.queue.clone();
            textures.color_texture_format = context.color_texture_format();
            textures.storage = WgpuAtlasStorage::default();
            textures.pending_uploads.clear();
        });
    }
}

impl PlatformAtlas for WgpuAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        self.0.lock().get_or_insert_with(key, build)
    }

    fn remove(&self, key: &AtlasKey) {
        self.0.lock().remove(key);
    }

    fn update_tile(
        &self,
        key: &AtlasKey,
        region: Bounds<DevicePixels>,
        bytes: &[u8],
        bytes_per_row: usize,
    ) -> bool {
        let mut lock = self.0.lock();
        let Some(tile) = lock.tile(key) else {
            return false;
        };
        let size = tile.bounds.size;
        let (x, y) = (region.origin.x.0, region.origin.y.0);
        let (w, h) = (region.size.width.0, region.size.height.0);
        if x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > size.width.0 || y + h > size.height.0 {
            return false;
        }
        let Some(texture) = lock.backend.storage.get(tile.texture_id) else {
            return false;
        };
        let bpp = texture.bytes_per_pixel() as usize;
        let row = w as usize * bpp;
        if bytes_per_row < row || bytes.len() < (h as usize - 1) * bytes_per_row + row {
            return false;
        }
        // Queued like an insert, so it lands after any upload still pending for this tile.
        let mut packed = Vec::with_capacity(row * h as usize);
        for r in 0..h as usize {
            packed.extend_from_slice(&bytes[r * bytes_per_row..r * bytes_per_row + row]);
        }
        let bounds = Bounds {
            origin: tile.bounds.origin + region.origin,
            size: region.size,
        };
        lock.backend.upload_texture(tile.texture_id, bounds, &packed);
        true
    }

    #[cfg(target_os = "linux")]
    fn copy_dmabuf_into_tile(&self, key: &AtlasKey, frame: &gpui::DmaBufFrame) -> bool {
        let mut lock = self.0.lock();
        let Some(tile) = lock.tile(key) else {
            return false;
        };
        let backend = &mut lock.backend;
        // Whatever an earlier insert or update still has queued for this texture lands first.
        backend.flush_uploads();
        let Some(texture) = backend.storage.get(tile.texture_id) else {
            return false;
        };
        match crate::dmabuf::copy_into(&backend.device, &backend.queue, frame, &texture.texture, tile.bounds) {
            Ok(()) => true,
            Err(error) => {
                log::warn!("dmabuf import failed: {error:#}");
                false
            }
        }
    }
}

impl AtlasBackend for WgpuAtlasTextures {
    fn insert(
        &mut self,
        kind: AtlasTextureKind,
        size: Size<DevicePixels>,
        bytes: &[u8],
    ) -> Result<AtlasTile> {
        let tile = self.allocate(size, kind).context("failed to allocate")?;
        self.upload_texture(tile.texture_id, tile.bounds, bytes);
        Ok(tile)
    }

    fn remove(&mut self, tile: AtlasTile) {
        let id = tile.texture_id;
        let Some(texture_slot) = self.storage[id.kind].textures.get_mut(id.index as usize) else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                self.pending_uploads
                    .retain(|upload| upload.id != texture.id);
                self.storage[id.kind]
                    .free_list
                    .push(texture.id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }
}

impl WgpuAtlasTextures {
    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = &mut self.storage[texture_kind];

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut WgpuAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        let max_texture_size = self.max_texture_size as i32;
        let max_atlas_size = Size {
            width: DevicePixels(max_texture_size),
            height: DevicePixels(max_texture_size),
        };

        let size = min_size.min(&max_atlas_size).max(&DEFAULT_ATLAS_SIZE);
        let format = match kind {
            AtlasTextureKind::Monochrome => wgpu::TextureFormat::R8Unorm,
            AtlasTextureKind::Subpixel | AtlasTextureKind::Polychrome => self.color_texture_format,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas"),
            size: wgpu::Extent3d {
                width: size.width.0 as u32,
                height: size.height.0 as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_list = &mut self.storage[kind];
        let index = texture_list.free_list.pop();

        let atlas_texture = WgpuAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            allocator: BucketedAtlasAllocator::new(device_size_to_etagere(size)),
            format,
            texture,
            view,
            live_atlas_keys: 0,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list
                .textures
                .get_mut(ix)
                .and_then(|t| t.as_mut())
                .expect("texture must exist")
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list
                .textures
                .last_mut()
                .and_then(|t| t.as_mut())
                .expect("texture must exist")
        }
    }

    fn upload_texture(&mut self, id: AtlasTextureId, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        let data = self
            .storage
            .get(id)
            .map(|texture| swizzle_upload_data(bytes, texture.format))
            .unwrap_or_else(|| bytes.to_vec());

        self.pending_uploads
            .push(PendingUpload { id, bounds, data });
    }

    fn flush_uploads(&mut self) {
        for upload in self.pending_uploads.drain(..) {
            let Some(texture) = self.storage.get(upload.id) else {
                continue;
            };
            let bytes_per_pixel = texture.bytes_per_pixel();

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: upload.bounds.origin.x.0 as u32,
                        y: upload.bounds.origin.y.0 as u32,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &upload.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.bounds.size.width.0 as u32 * bytes_per_pixel as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: upload.bounds.size.width.0 as u32,
                    height: upload.bounds.size.height.0 as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

#[derive(Default)]
struct WgpuAtlasStorage {
    monochrome_textures: AtlasTextureList<WgpuAtlasTexture>,
    subpixel_textures: AtlasTextureList<WgpuAtlasTexture>,
    polychrome_textures: AtlasTextureList<WgpuAtlasTexture>,
}

impl ops::Index<AtlasTextureKind> for WgpuAtlasStorage {
    type Output = AtlasTextureList<WgpuAtlasTexture>;
    fn index(&self, kind: AtlasTextureKind) -> &Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Subpixel => &self.subpixel_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        }
    }
}

impl ops::IndexMut<AtlasTextureKind> for WgpuAtlasStorage {
    fn index_mut(&mut self, kind: AtlasTextureKind) -> &mut Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
        }
    }
}

impl WgpuAtlasStorage {
    fn get(&self, id: AtlasTextureId) -> Option<&WgpuAtlasTexture> {
        self[id.kind]
            .textures
            .get(id.index as usize)
            .and_then(|t| t.as_ref())
    }
}

impl ops::Index<AtlasTextureId> for WgpuAtlasStorage {
    type Output = WgpuAtlasTexture;
    fn index(&self, id: AtlasTextureId) -> &Self::Output {
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Subpixel => &self.subpixel_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        };
        textures[id.index as usize]
            .as_ref()
            .expect("texture must exist")
    }
}

struct WgpuAtlasTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    live_atlas_keys: u32,
}

impl WgpuAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(device_size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            padding: 0,
            bounds: Bounds {
                origin: etagere_point_to_device(allocation.rectangle.min),
                size,
            },
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn bytes_per_pixel(&self) -> u8 {
        match self.format {
            wgpu::TextureFormat::R8Unorm => 1,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm => 4,
            _ => 4,
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn swizzle_upload_data(bytes: &[u8], format: wgpu::TextureFormat) -> Vec<u8> {
    match format {
        wgpu::TextureFormat::Rgba8Unorm => {
            let mut data = bytes.to_vec();
            for pixel in data.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
            data
        }
        _ => bytes.to_vec(),
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use gpui::block_on;
    use gpui::{ImageId, RenderImageParams};
    use std::sync::Arc;

    fn test_device_and_queue() -> anyhow::Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::default(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|error| anyhow::anyhow!("failed to request adapter: {error}"))?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("wgpu_atlas_test_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults()
                        .using_resolution(adapter.limits())
                        .using_alignment(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                })
                .await
                .map_err(|error| anyhow::anyhow!("failed to request device: {error}"))?;
            Ok((Arc::new(device), Arc::new(queue)))
        })
    }

    #[test]
    fn update_tile_queues_only_the_region() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        let key = AtlasKey::Image(RenderImageParams {
            image_id: ImageId(7),
            frame_index: 0,
        });
        let size = Size {
            width: DevicePixels(4),
            height: DevicePixels(4),
        };
        let region = |x, y, w, h| Bounds {
            origin: Point::new(DevicePixels(x), DevicePixels(y)),
            size: Size::new(DevicePixels(w), DevicePixels(h)),
        };
        // a 4x4 frame whose bytes are their own index
        let frame: Vec<u8> = (0..64).collect();
        assert!(!atlas.update_tile(&key, region(0, 0, 1, 1), &frame, 16));

        let tile = atlas
            .get_or_insert_with(key.clone(), &mut || Ok(Some((size, Cow::Owned(vec![0; 64])))))?
            .expect("tile should be created");
        atlas.before_frame();

        // rows 1-2, columns 1-2, read from the frame at its own row pitch
        assert!(atlas.update_tile(&key, region(1, 1, 2, 2), &frame[16 + 4..], 16));
        {
            let lock = atlas.0.lock();
            let upload = lock.backend.pending_uploads.last().expect("an upload");
            assert_eq!(upload.bounds, Bounds {
                origin: tile.bounds.origin + Point::new(DevicePixels(1), DevicePixels(1)),
                size: Size::new(DevicePixels(2), DevicePixels(2)),
            });
            assert_eq!(upload.data, [20, 21, 22, 23, 24, 25, 26, 27, 36, 37, 38, 39, 40, 41, 42, 43]);
        }
        // past the tile, or too few bytes for the rows asked
        assert!(!atlas.update_tile(&key, region(3, 3, 2, 2), &frame, 16));
        assert!(!atlas.update_tile(&key, region(0, 0, 4, 4), &frame[..40], 16));
        atlas.before_frame();
        Ok(())
    }

    #[test]
    fn before_frame_skips_uploads_for_removed_texture() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;

        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);
        let key = AtlasKey::Image(RenderImageParams {
            image_id: ImageId(1),
            frame_index: 0,
        });
        let size = Size {
            width: DevicePixels(1),
            height: DevicePixels(1),
        };
        let mut build = || Ok(Some((size, Cow::Owned(vec![0, 0, 0, 255]))));

        // Regression test: before the fix, this panicked in flush_uploads
        atlas
            .get_or_insert_with(key.clone(), &mut build)?
            .expect("tile should be created");
        atlas.remove(&key);
        atlas.before_frame();
        Ok(())
    }

    #[test]
    fn remove_deallocates_tile_space_for_reuse() -> anyhow::Result<()> {
        let (device, queue) = test_device_and_queue()?;
        let atlas = WgpuAtlas::new(device, queue, wgpu::TextureFormat::Bgra8Unorm);

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let big = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };

        let make_key = |image_id: usize| {
            AtlasKey::Image(RenderImageParams {
                image_id: ImageId(image_id),
                frame_index: 0,
            })
        };
        let insert = |key: AtlasKey, size: Size<DevicePixels>| {
            let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
            atlas
                .get_or_insert_with(key, &mut || {
                    Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
                })
                .expect("allocation should succeed")
                .expect("callback returns Some")
        };

        let keeper_key = make_key(1);
        let big_key_a = make_key(2);
        let big_key_b = make_key(3);

        let keeper_tile = insert(keeper_key, small);
        let tile_a = insert(big_key_a.clone(), big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);
        let tile_b = insert(big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
        Ok(())
    }

    #[test]
    fn swizzle_upload_data_preserves_bgra_uploads() {
        let input = vec![0x10, 0x20, 0x30, 0x40];
        assert_eq!(
            swizzle_upload_data(&input, wgpu::TextureFormat::Bgra8Unorm),
            input
        );
    }

    #[test]
    fn swizzle_upload_data_converts_bgra_to_rgba() {
        let input = vec![0x10, 0x20, 0x30, 0x40, 0xAA, 0xBB, 0xCC, 0xDD];
        assert_eq!(
            swizzle_upload_data(&input, wgpu::TextureFormat::Rgba8Unorm),
            vec![0x30, 0x20, 0x10, 0x40, 0xCC, 0xBB, 0xAA, 0xDD]
        );
    }
}
