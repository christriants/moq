use crate as hang;
use anyhow::Context;
use buf_list::BufList;
use bytes::{Buf, Bytes};
use moq_lite as moq;
use scuffle_av1::seq::SequenceHeaderObu;

/// A decoder for AV1 with inline sequence headers.
pub struct Av01 {
	// The broadcast being produced.
	broadcast: hang::BroadcastProducer,

	// The track being produced.
	track: Option<hang::TrackProducer>,

	// Whether the track has been initialized.
	config: Option<hang::catalog::VideoConfig>,

	// The current frame being built.
	current: Frame,

	// Used to compute wall clock timestamps if needed.
	zero: Option<tokio::time::Instant>,

	// Track if we've sent the first frame yet
	first_frame_sent: bool,
}

#[derive(Default)]
struct Frame {
	chunks: BufList,
	contains_keyframe: bool,
	contains_frame: bool,
}

impl Av01 {
	pub fn new(broadcast: hang::BroadcastProducer) -> Self {
		Self {
			broadcast,
			track: None,
			config: None,
			current: Default::default(),
			zero: None,
			first_frame_sent: false,
		}
	}

	fn setup_track(&mut self, seq_header: &SequenceHeaderObu) -> anyhow::Result<()> {
		let config = create_av1_config(
			seq_header.seq_profile,
			seq_header.operating_points.first().map(|op| op.seq_level_idx).unwrap_or(0),
			if seq_header.operating_points.first().map(|op| op.seq_tier).unwrap_or(false) { 'H' } else { 'M' },
			seq_header.color_config.bit_depth as u8,
			seq_header.color_config.mono_chrome,
			seq_header.color_config.subsampling_x,
			seq_header.color_config.subsampling_y,
			seq_header.color_config.chroma_sample_position,
			seq_header.color_config.color_primaries,
			seq_header.color_config.transfer_characteristics,
			seq_header.color_config.matrix_coefficients,
			seq_header.color_config.full_color_range,
			Some(seq_header.max_frame_width as u32),
			Some(seq_header.max_frame_height as u32),
		);

		if let Some(old) = &self.config {
			if old == &config {
				return Ok(());
			}
		}

		if let Some(track) = &self.track.take() {
			tracing::debug!(name = ?track.info.name, "reinitializing track");
			self.broadcast.catalog.lock().remove_video(&track.info.name);
		}

		let track = moq::Track {
			name: self.broadcast.track_name("video"),
			priority: 2,
		};

		tracing::debug!(name = ?track.name, ?config, "starting track");

		{
			let mut catalog = self.broadcast.catalog.lock();
			let video = catalog.insert_video(track.name.clone(), config.clone());
			video.priority = 2;
		}

		let track = track.produce();
		self.broadcast.insert_track(track.consumer);

		self.config = Some(config);
		self.track = Some(track.producer.into());

		Ok(())
	}

	/// Initialize the decoder with sequence header and other metadata OBUs.
	pub fn initialize<T: Buf + AsRef<[u8]>>(&mut self, buf: &mut T) -> anyhow::Result<()> {
		let data = buf.as_ref();

		tracing::debug!(
			"initialize called with {} bytes: {:02x?}",
			data.len(),
			&data[..std::cmp::min(16, data.len())]
		);

		if data.len() >= 4 && data[0] == 0x0a {
			tracing::debug!("Parsing av1C box - extracting config directly");

			let seq_profile = (data[1] >> 5) & 0x07;
			let seq_level_idx = data[1] & 0x1F;
			let tier = (data[2] >> 7) & 0x01;
			let high_bitdepth = (data[2] >> 6) & 0x01;
			let twelve_bit = (data[2] >> 5) & 0x01;

			let bitdepth = if high_bitdepth == 1 {
				if twelve_bit == 1 { 12 } else { 10 }
			} else {
				8
			};

			// Create config directly from av1C header
			let config = create_av1_config(
				seq_profile,
				seq_level_idx,
				if tier == 1 { 'H' } else { 'M' },
				bitdepth,
				((data[2] >> 4) & 0x01) == 1,
				((data[2] >> 3) & 0x01) == 1,
				((data[2] >> 2) & 0x01) == 1,
				data[2] & 0x03,
				1, // BT.709
				1, // BT.709
				1, // BT.709
				false, // av1C doesn't provide this
				None, // Let browser determine
				None,
			);

			// Initialize track with this config
			// @todo add dimensions to name
			let track = moq::Track {
				name: self.broadcast.track_name("video"),
				priority: 2,
			};

			tracing::debug!(name = ?track.name, ?config, "starting track from av1C");

			{
				let mut catalog = self.broadcast.catalog.lock();
				let video = catalog.insert_video(track.name.clone(), config.clone());
				video.priority = 2;
			}

			let track = track.produce();
			self.broadcast.insert_track(track.consumer);

			self.config = Some(config);
			self.track = Some(track.producer.into());

			buf.advance(data.len());
			return Ok(());
		}

		// Parse raw OBUs for other formats
		let mut obus = ObuIterator::new(buf);

		while let Some(obu) = obus.next().transpose()? {
			self.decode_obu(obu, None)?;
		}

		if let Some(obu) = obus.flush()? {
			self.decode_obu(obu, None)?;
		}

		Ok(())
	}

	/// Decode as much data as possible from the given buffer.
	pub fn decode_stream<T: Buf + AsRef<[u8]>>(
		&mut self,
		buf: &mut T,
		pts: Option<hang::Timestamp>,
	) -> anyhow::Result<()> {
		let pts = self.pts(pts)?;

		let obus = ObuIterator::new(buf);

		for obu in obus {
			self.decode_obu(obu?, Some(pts))?;
		}

		Ok(())
	}

	/// Decode all data in the buffer, assuming the buffer contains (the rest of) a frame.
	pub fn decode_frame<T: Buf + AsRef<[u8]>>(
		&mut self,
		buf: &mut T,
		pts: Option<hang::Timestamp>,
	) -> anyhow::Result<()> {
		let pts = self.pts(pts)?;
		let mut obus = ObuIterator::new(buf);

		while let Some(obu) = obus.next().transpose()? {
			self.decode_obu(obu, Some(pts))?;
		}

		if let Some(obu) = obus.flush()? {
			self.decode_obu(obu, Some(pts))?;
		}

		self.maybe_start_frame(Some(pts))?;

		Ok(())
	}
	fn decode_obu(&mut self, obu_data: Bytes, pts: Option<hang::Timestamp>) -> anyhow::Result<()> {
		anyhow::ensure!(obu_data.len() >= 1, "OBU is too short");

		// Parse OBU header
		let header = scuffle_av1::ObuHeader::parse(&mut &obu_data[..])?;

		// Match on the ObuType enum directly
		use scuffle_av1::ObuType;
		match header.obu_type {
			ObuType::SequenceHeader => {
				// Try to parse, but if it fails (broken OBS headers), just skip parsing
				match SequenceHeaderObu::parse(header, &mut &obu_data[1..]) {
					Ok(seq_header) => {
						self.setup_track(&seq_header)?;
					}
					Err(e) => {
						tracing::warn!("Failed to parse sequence header OBU, skipping: {}", e);

					}
				}

				// Still include the (possibly broken) sequence header in the frame data
				// The browser might be able to handle it even if our parser can't
			}
			ObuType::TemporalDelimiter => {
				self.maybe_start_frame(pts)?;
			}
			ObuType::FrameHeader | ObuType::Frame => {
				// Force the first frame to be a keyframe
				let is_first_frame = !self.first_frame_sent;

				if is_first_frame {
					self.first_frame_sent = true;
				}

				// Check if this is a keyframe by looking at the frame header
				let is_keyframe = is_first_frame || {
					if obu_data.len() > 1 {
						(obu_data[1] >> 7) & 1 == 0
					} else {
						false
					}
				};

				if is_keyframe {
					self.current.contains_keyframe = true;
				}
				self.current.contains_frame = true;
			}
			ObuType::Metadata => {
				self.maybe_start_frame(pts)?;
			}
			ObuType::TileGroup | ObuType::TileList => {
				self.current.contains_frame = true;
			}
			_ => {
				// Other OBU types - just include them
			}
		}

		tracing::trace!(?header.obu_type, "parsed OBU");

		self.current.chunks.push_chunk(obu_data);

		Ok(())
	}
	fn maybe_start_frame(&mut self, pts: Option<hang::Timestamp>) -> anyhow::Result<()> {
		if !self.current.contains_frame {
			return Ok(());
		}

		let track = self.track.as_mut().context("expected sequence header before any frames")?;
		let pts = pts.context("missing timestamp")?;

		let payload = std::mem::take(&mut self.current.chunks);
		let frame = hang::Frame {
			timestamp: pts,
			keyframe: self.current.contains_keyframe,
			payload,
		};

		track.write(frame)?;

		self.current.contains_keyframe = false;
		self.current.contains_frame = false;

		Ok(())
	}

	pub fn is_initialized(&self) -> bool {
		self.track.is_some()
	}

	fn pts(&mut self, hint: Option<hang::Timestamp>) -> anyhow::Result<hang::Timestamp> {
		if let Some(pts) = hint {
			return Ok(pts);
		}

		let zero = self.zero.get_or_insert_with(tokio::time::Instant::now);
		Ok(hang::Timestamp::from_micros(zero.elapsed().as_micros() as u64)?)
	}
}

impl Drop for Av01 {
	fn drop(&mut self) {
		if let Some(track) = &self.track {
			tracing::debug!(name = ?track.info.name, "ending track");
			self.broadcast.catalog.lock().remove_video(&track.info.name);
		}
	}
}

/// Iterator over AV1 Open Bitstream Units (OBUs)
struct ObuIterator<'a, T: Buf + AsRef<[u8]> + 'a> {
	buf: &'a mut T,
}

impl<'a, T: Buf + AsRef<[u8]> + 'a> ObuIterator<'a, T> {
	pub fn new(buf: &'a mut T) -> Self {
		Self { buf }
	}

	pub fn flush(self) -> anyhow::Result<Option<Bytes>> {
		let remaining = self.buf.remaining();
		if remaining == 0 {
			return Ok(None);
		}

		let obu = self.buf.copy_to_bytes(remaining);
		Ok(Some(obu))
	}
}

impl<'a, T: Buf + AsRef<[u8]> + 'a> Iterator for ObuIterator<'a, T> {
	type Item = anyhow::Result<Bytes>;

	fn next(&mut self) -> Option<Self::Item> {
		if self.buf.remaining() == 0 {
			return None;
		}

		// Parse OBU header to get size
		let data = self.buf.as_ref();
		if data.is_empty() {
			return None;
		}

		// OBU header format:
		// - obu_forbidden_bit (1)
		// - obu_type (4)
		// - obu_extension_flag (1)
		// - obu_has_size_field (1)
		// - obu_reserved_1bit (1)

		let header = data[0];
		let has_size = (header >> 1) & 1 == 1;

		if !has_size {
			// Without size field, consume entire buffer as one OBU
			let remaining = self.buf.remaining();
			let obu = self.buf.copy_to_bytes(remaining);
			return Some(Ok(obu));
		}

		// Parse LEB128 size
		let mut size: usize = 0;
		let mut offset = 1;
		let mut shift = 0;

		loop {
			if offset >= data.len() {
				return None; // Need more data
			}

			let byte = data[offset];
			offset += 1;

			size |= ((byte & 0x7F) as usize) << shift;
			shift += 7;

			if byte & 0x80 == 0 {
				break;
			}

			if shift >= 56 {
				return Some(Err(anyhow::anyhow!("OBU size too large")));
			}
		}

		let total_size = offset + size;

		if total_size > self.buf.remaining() {
			// Need more data
			return None;
		}

		let obu = self.buf.copy_to_bytes(total_size);
		Some(Ok(obu))
	}
}

fn create_av1_config(
    profile: u8,
    level: u8,
    tier: char,
    bitdepth: u8,
    mono_chrome: bool,
    chroma_subsampling_x: bool,
    chroma_subsampling_y: bool,
    chroma_sample_position: u8,
    color_primaries: u8,
    transfer_characteristics: u8,
    matrix_coefficients: u8,
    full_range: bool,
    coded_width: Option<u32>,
    coded_height: Option<u32>,
) -> hang::catalog::VideoConfig {
    hang::catalog::VideoConfig {
        coded_width,
        coded_height,
        codec: hang::catalog::AV1 {
            profile,
            level,
            tier,
            bitdepth,
            mono_chrome,
            chroma_subsampling_x,
            chroma_subsampling_y,
            chroma_sample_position,
            color_primaries,
            transfer_characteristics,
            matrix_coefficients,
            full_range,
        }.into(),
        description: None,
        framerate: None,
        bitrate: None,
        display_ratio_width: None,
        display_ratio_height: None,
        optimize_for_latency: None,
    }
}
