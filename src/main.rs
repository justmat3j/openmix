mod audio;
mod export;
mod metadata;
mod model;
mod planning;
mod storage;
mod theme;
mod ui;

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fs::{self, File},
    path::{Path, PathBuf},
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use eframe::{
    egui,
    egui::{Align, Color32, CornerRadius, Frame, Layout, Margin, RichText, Stroke, Vec2},
};
use export::*;
use metadata::*;
use model::*;
use planning::*;
use rodio::{Decoder, DeviceSinkBuilder, MixerDeviceSink, Player, Source, buffer::SamplesBuffer};
use serde::{Deserialize, Serialize};
use storage::*;
use theme::*;
use ui::*;

// "Mix all" streams the render straight to a WAV file on disk (see
// audio::MixWriter) instead of holding the whole thing in memory, so these
// caps are just sanity limits rather than memory guards: the track count
// keeps a single render from taking forever, and the duration cap keeps it
// from filling the disk.
const MIX_ALL_MAX_TRACKS: usize = 60;
const MIX_ALL_MAX_DURATION: Duration = Duration::from_secs(24 * 60 * 60);

thread_local! {
    static COVER_TEXTURES: RefCell<HashMap<PathBuf, egui::TextureHandle>> = RefCell::new(HashMap::new());
}

struct AudioEngine {
    stream: Option<MixerDeviceSink>,
    player: Option<Player>,
    duration: Duration,
    error: Option<String>,
    volume: f32,
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self {
            stream: None,
            player: None,
            duration: Duration::ZERO,
            error: None,
            volume: 0.8,
        }
    }
}

impl AudioEngine {
    fn ensure_stream(&mut self) -> Result<(), String> {
        if self.stream.is_none() {
            self.stream = Some(
                DeviceSinkBuilder::open_default_sink()
                    .map_err(|e| format!("No audio output is available: {e}"))?,
            );
        }
        Ok(())
    }

    fn new_player(&mut self) -> Result<Player, String> {
        self.ensure_stream()?;
        let player = Player::connect_new(self.stream.as_ref().unwrap().mixer());
        player.set_volume(self.volume);
        Ok(player)
    }

    fn play_file(&mut self, path: &Path, known_duration: u32) -> Result<Duration, String> {
        let file = File::open(path).map_err(|e| format!("Could not open this track: {e}"))?;
        let decoder =
            Decoder::try_from(file).map_err(|e| format!("Could not decode this track: {e}"))?;
        let duration = decoder
            .total_duration()
            .unwrap_or(Duration::from_secs(known_duration as u64));
        if let Some(old) = self.player.take() {
            old.stop();
        }
        let player = self.new_player()?;
        player.append(decoder);
        player.play();
        self.duration = duration;
        self.player = Some(player);
        self.error = None;
        Ok(duration)
    }

    /// Plays a "mix all" render streamed from disk (see `mix_all`), rather
    /// than loading the whole rendered mix into memory the way
    /// `play_pcm_from` does for the short two-track preview.
    fn play_mix_file(&mut self, path: &Path, duration: Duration) -> Result<(), String> {
        let file = File::open(path).map_err(|e| format!("Could not open the mix: {e}"))?;
        let decoder =
            Decoder::try_from(file).map_err(|e| format!("Could not decode the mix: {e}"))?;
        if let Some(old) = self.player.take() {
            old.stop();
        }
        let player = self.new_player()?;
        player.append(decoder);
        player.play();
        self.duration = duration;
        self.player = Some(player);
        self.error = None;
        Ok(())
    }

    fn play_pcm_from(&mut self, pcm: audio::StereoPcm, position: Duration) -> Result<(), String> {
        if let Some(old) = self.player.take() {
            old.stop();
        }
        let player = self.new_player()?;
        let channels = std::num::NonZeroU16::new(2).unwrap();
        let rate = std::num::NonZeroU32::new(audio::OUTPUT_RATE).unwrap();
        player.append(SamplesBuffer::new(channels, rate, pcm.samples));
        player.pause();
        self.duration = pcm.duration;
        self.player = Some(player);
        if position > Duration::ZERO {
            self.player
                .as_ref()
                .unwrap()
                .try_seek(position.min(self.duration))
                .map_err(|e| format!("Could not align the prepared mix: {e}"))?;
        }
        self.player.as_ref().unwrap().play();
        self.error = None;
        Ok(())
    }

    fn toggle(&mut self) -> Result<(), String> {
        let player = self
            .player
            .as_ref()
            .ok_or_else(|| "Choose a track to start listening.".to_owned())?;
        if player.is_paused() {
            player.play();
        } else {
            player.pause();
        }
        Ok(())
    }

    fn position(&self) -> Duration {
        self.player
            .as_ref()
            .map(Player::get_pos)
            .unwrap_or_default()
    }
    fn is_playing(&self) -> bool {
        self.player
            .as_ref()
            .is_some_and(|p| !p.is_paused() && !p.empty())
    }
    fn ended(&self) -> bool {
        self.player.as_ref().is_some_and(Player::empty)
    }
    fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
        if let Some(player) = &self.player {
            player.set_volume(self.volume);
        }
    }
    fn seek(&mut self, position: Duration) -> Result<(), String> {
        let player = self
            .player
            .as_ref()
            .ok_or_else(|| "Choose a track before seeking.".to_owned())?;
        player
            .try_seek(position.min(self.duration))
            .map_err(|e| format!("Could not seek: {e}"))
    }
}

struct OpenMix {
    data: LibraryData,
    page: Page,
    active_playlist: Option<String>,
    search: String,
    sort: SortMode,
    sort_ascending: bool,
    queue: Vec<String>,
    current: Option<String>,
    shuffle: bool,
    repeat: u8,
    now_open: bool,
    search_focus: bool,
    status: String,
    toast: Option<Toast>,
    key_input: String,
    key_status: String,
    key_visible: bool,
    animated_page: Page,
    page_animation_target: bool,
    notes: HashMap<String, (String, bool)>,
    playlist_dialog: bool,
    playlist_edit: Option<String>,
    playlist_name: String,
    playlist_delete: Option<String>,
    selected_tracks: HashSet<String>,
    audio: AudioEngine,
    muted: bool,
    volume_before_mute: f32,
    config_file: PathBuf,
    tx: Sender<WorkerMessage>,
    rx: Receiver<WorkerMessage>,
    /// Path to the most recently generated mix source on disk. When this is
    /// set, the player bar exposes an "Export mix" button.
    last_mix_path: Option<PathBuf>,
}

impl OpenMix {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Public MusicBrainz metadata does not use a key; remove any legacy saved credential.
        let _ = clear_secret();
        let config_file = config_path();
        let data = load_library(&config_file).unwrap_or_default();
        set_theme(&cc.egui_ctx, data.theme);
        let key = read_secret()
            .or_else(|| std::env::var("NVIDIA_API_KEY").ok())
            .unwrap_or_default();
        let (tx, rx) = mpsc::channel();
        Self {
            data,
            page: Page::Listen,
            active_playlist: None,
            search: String::new(),
            sort: SortMode::RecentlyAdded,
            sort_ascending: false,
            queue: Vec::new(),
            current: None,
            shuffle: false,
            repeat: 0,
            now_open: true,
            search_focus: false,
            status: String::new(),
            toast: None,
            key_input: key,
            key_status:
                "The API key is kept in your system credential store when available (NVIDIA NIM)."
                    .into(),
            key_visible: false,
            animated_page: Page::Listen,
            page_animation_target: true,
            notes: HashMap::new(),
            playlist_dialog: false,
            playlist_edit: None,
            playlist_name: String::new(),
            playlist_delete: None,
            selected_tracks: HashSet::new(),
            audio: AudioEngine::default(),
            muted: false,
            volume_before_mute: 0.8,
            config_file,
            tx,
            rx,
            last_mix_path: None,
        }
    }

    fn persist(&mut self) {
        match save_library(&self.config_file, &self.data) {
            Ok(()) => {}
            Err(error) => self.status = format!("Could not save library: {error}"),
        }
    }

    fn change_theme(&mut self, ctx: &egui::Context, theme: AppTheme) {
        if self.data.theme == theme {
            return;
        }
        self.data.theme = theme;
        set_theme(ctx, theme);
        self.persist();
        self.notify(
            "Theme updated",
            format!("{} is now active.", theme.label()),
            false,
        );
    }

    fn notify(&mut self, title: impl Into<String>, message: impl Into<String>, error: bool) {
        self.toast = Some(Toast {
            title: title.into(),
            message: message.into(),
            error,
            expires_at: Instant::now() + Duration::from_secs(5),
        });
    }

    fn lookup_metadata(&mut self, tracks: Vec<Track>) {
        if tracks.is_empty() {
            self.notify("MusicBrainz", "There are no tracks to look up.", false);
            return;
        }
        self.notify(
            "MusicBrainz",
            format!("Looking up metadata for {} track(s)…", tracks.len()),
            false,
        );
        let tx = self.tx.clone();
        thread::spawn(move || {
            let total = tracks.len();
            let mut matched = 0;
            let mut misses = 0;
            let mut unmatched = Vec::new();
            let mut errors = Vec::new();
            for track in tracks {
                let result = musicbrainz_metadata(&track);
                match &result {
                    Ok(Some(_)) => matched += 1,
                    Ok(None) => {
                        misses += 1;
                        unmatched.push(format!("{} — {}", track.title, track.artist));
                    }
                    Err(error) => errors.push(format!("{}: {error}", track.title)),
                }
                let _ = tx.send(WorkerMessage::MetadataFetched {
                    track_id: track.id,
                    result,
                });
            }
            let _ = tx.send(WorkerMessage::MetadataFetchFinished {
                total,
                matched,
                misses,
                unmatched,
                errors,
            });
        });
    }

    fn import_files(&mut self) {
        let Some(files) = rfd::FileDialog::new()
            .add_filter("Audio", &["mp3", "wav"])
            .pick_files()
        else {
            return;
        };
        if files.is_empty() {
            return;
        }
        self.status = format!("Analyzing {} file(s)…", files.len());
        let tx = self.tx.clone();
        thread::spawn(move || {
            let mut imported = Vec::new();
            let mut problems = Vec::new();
            for path in files {
                match audio::analyze_file(&path) {
                    Ok(analysis) => {
                        let filename = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("Local file");
                        let (filename_artist, filename_title) = filename
                            .split_once(" - ")
                            .map(|(artist, title)| {
                                (artist.trim().to_owned(), title.trim().to_owned())
                            })
                            .unwrap_or_else(|| ("Local file".into(), filename.to_owned()));
                        let artist = analysis.artist.unwrap_or(filename_artist);
                        let title = analysis.title.unwrap_or(filename_title);
                        let path_string = path.to_string_lossy().into_owned();
                        imported.push(Track {
                            id: path_string.clone(),
                            path,
                            title,
                            artist,
                            album: analysis.album.unwrap_or_else(|| "Local library".into()),
                            release_year: None,
                            musicbrainz_id: None,
                            cover_art_path: None,
                            duration: analysis.duration_seconds,
                            bpm: analysis.bpm,
                            energy: analysis.energy,
                            rise: analysis.rise_seconds,
                            added: now_seconds(),
                            last_played: 0,
                            liked: false,
                        });
                    }
                    Err(error) => problems.push(format!(
                        "{}: {error}",
                        path.file_name().unwrap_or_default().to_string_lossy()
                    )),
                }
            }
            let result = if imported.is_empty() {
                Err(problems.join("\n"))
            } else {
                Ok((imported, problems))
            };
            let _ = tx.send(WorkerMessage::Imported(result));
        });
    }

    fn add_to_queue(&mut self, id: &str) {
        if self.data.tracks.iter().any(|t| t.id == id) {
            self.queue.push(id.to_owned());
            self.status = format!("Added to queue · {} up next", self.queue.len());
        }
    }

    fn play_id(&mut self, id: &str) {
        let Some(track) = self.data.tracks.iter().find(|t| t.id == id).cloned() else {
            return;
        };
        match self.audio.play_file(&track.path, track.duration) {
            Ok(_) => {
                self.current = Some(track.id.clone());
                self.data
                    .tracks
                    .iter_mut()
                    .find(|t| t.id == track.id)
                    .unwrap()
                    .last_played = now_seconds();
                self.data.recent.retain(|item| item != &track.id);
                self.data.recent.insert(0, track.id);
                self.data.recent.truncate(50);
                self.persist();
                self.status.clear();
            }
            Err(error) => {
                self.audio.error = Some(error.clone());
                self.status = error;
            }
        }
    }

    fn toggle(&mut self) {
        if self.current.is_none() {
            if let Some(next) = self.queue.first().cloned() {
                self.queue.remove(0);
                self.play_id(&next);
            } else if let Some(first) = self.data.tracks.first().map(|t| t.id.clone()) {
                self.play_id(&first);
            }
            return;
        }
        if let Err(error) = self.audio.toggle() {
            self.status = error;
        }
    }

    fn next(&mut self) {
        if self.repeat == 1 {
            if let Some(current) = self.current.clone() {
                self.play_id(&current);
            }
            return;
        }
        if !self.queue.is_empty() {
            let next = self.queue.remove(0);
            self.play_id(&next);
            return;
        }
        if self.data.tracks.is_empty() {
            return;
        }
        let ids = if let Some(playlist_id) = self.active_playlist.as_deref() {
            self.data
                .playlists
                .iter()
                .find(|p| p.id == playlist_id)
                .map(|p| p.tracks.clone())
                .unwrap_or_default()
        } else {
            self.visible_tracks()
                .into_iter()
                .map(|t| t.id.clone())
                .collect()
        };
        if ids.is_empty() {
            return;
        }
        let index = self
            .current
            .as_ref()
            .and_then(|id| ids.iter().position(|item| item == id))
            .unwrap_or(0);
        let next = if self.shuffle && ids.len() > 1 {
            let mut n = (now_seconds() as usize ^ index.wrapping_mul(31)) % ids.len();
            if ids[n] == self.current.clone().unwrap_or_default() {
                n = (n + 1) % ids.len();
            }
            n
        } else if index + 1 < ids.len() {
            index + 1
        } else if self.repeat == 2 {
            0
        } else {
            return;
        };
        self.play_id(&ids[next]);
    }

    fn previous(&mut self) {
        if self.audio.position() > Duration::from_secs(3) {
            let _ = self.audio.seek(Duration::ZERO);
            return;
        }
        let ids: Vec<String> = self
            .visible_tracks()
            .into_iter()
            .map(|t| t.id.clone())
            .collect();
        let index = self
            .current
            .as_ref()
            .and_then(|id| ids.iter().position(|item| item == id))
            .unwrap_or(0);
        if index > 0 {
            self.play_id(&ids[index - 1]);
        }
    }

    fn current_track(&self) -> Option<Track> {
        self.current
            .as_ref()
            .and_then(|id| self.data.tracks.iter().find(|t| &t.id == id))
            .cloned()
    }

    fn visible_tracks(&self) -> Vec<Track> {
        let mut tracks: Vec<Track> = match self.page {
            Page::Liked => self
                .data
                .tracks
                .iter()
                .filter(|t| t.liked)
                .cloned()
                .collect(),
            Page::Playlist => {
                let ids = self
                    .data
                    .playlists
                    .iter()
                    .find(|p| Some(p.id.as_str()) == self.active_playlist.as_deref())
                    .map(|p| p.tracks.as_slice())
                    .unwrap_or(&[]);
                ids.iter()
                    .filter_map(|id| {
                        self.data
                            .tracks
                            .iter()
                            .find(|track| &track.id == id)
                            .cloned()
                    })
                    .collect()
            }
            _ => self.data.tracks.clone(),
        };
        let query = self.search.trim().to_lowercase();
        if !query.is_empty() {
            tracks.retain(|t| {
                format!("{} {} {} {}", t.title, t.artist, t.album, t.path.display())
                    .to_lowercase()
                    .contains(&query)
            });
        }
        match self.sort {
            SortMode::RecentlyAdded => tracks.sort_by_key(|t| t.added),
            SortMode::PlaylistOrder => {}
            SortMode::Title => tracks.sort_by_cached_key(|track| track.title.to_lowercase()),
            SortMode::Artist => tracks.sort_by(|a, b| {
                a.artist
                    .to_lowercase()
                    .cmp(&b.artist.to_lowercase())
                    .then(a.title.to_lowercase().cmp(&b.title.to_lowercase()))
            }),
            SortMode::Duration => tracks.sort_by_key(|t| t.duration),
            SortMode::Bpm => tracks.sort_by_key(|t| t.bpm),
        }
        if !self.sort_ascending {
            tracks.reverse();
        }
        tracks
    }

    fn queue_transition(&mut self, outgoing: Track, incoming: Track) {
        let outgoing_position = if self.current.as_deref() == Some(outgoing.id.as_str()) {
            self.audio.position().as_secs().min(u64::from(u32::MAX)) as u32
        } else {
            0
        };
        let key = self.key_input.trim().to_owned();
        let use_openrouter = self.data.use_openrouter && !key.is_empty();
        self.status = if key.is_empty() {
            "Preparing a local mix…".into()
        } else if !self.data.use_openrouter {
            "Preparing a local mix… (AI mixing is off)".into()
        } else {
            "Planning your mix…".into()
        };
        let tx = self.tx.clone();
        thread::spawn(move || {
            let fallback_plan = local_plan_at(&outgoing, &incoming, outgoing_position);
            let (plan, rationale, used_local_plan) = if !use_openrouter {
                (fallback_plan, local_note(&outgoing, &incoming), true)
            } else {
                match openrouter_plan(&key, &outgoing, &incoming, outgoing_position) {
                    Ok((plan, rationale)) => (plan, rationale, false),
                    Err(error) => (fallback_plan, format!("Local plan used. {error}"), true),
                }
            };
            let rationale = format!(
                "{} {}s lead, {}s fade, incoming at {}s.",
                rationale, plan.entry_lead_seconds, plan.fade_seconds, plan.incoming_start_seconds,
            );
            let result = audio::decode_stereo(&outgoing.path).and_then(|a| {
                audio::decode_stereo(&incoming.path).map(|b| {
                    (
                        a,
                        b,
                        plan,
                        rationale,
                        used_local_plan,
                        outgoing.bpm,
                        incoming.bpm,
                    )
                })
            });
            let _ = tx.send(WorkerMessage::TransitionDecoded {
                outgoing_id: outgoing.id,
                incoming_id: incoming.id,
                result,
            });
        });
    }

    fn mix_all(&mut self, tracks: Vec<Track>) {
        if tracks.len() < 2 {
            if let Some(track) = tracks.first() {
                self.play_id(&track.id);
            }
            return;
        }
        if tracks.len() > MIX_ALL_MAX_TRACKS {
            self.status = format!(
                "Mix all is limited to {MIX_ALL_MAX_TRACKS} tracks per render to keep memory use predictable. This selection has {}.",
                tracks.len()
            );
            return;
        }
        self.status = format!("Preparing a continuous mix of {} tracks…", tracks.len());
        let key = self.key_input.trim().to_owned();
        let use_openrouter = self.data.use_openrouter && !key.is_empty();
        let first_id = tracks[0].id.clone();
        let tx = self.tx.clone();
        thread::spawn(move || {
            let result = (|| {
                let track_ids: Vec<String> = tracks.iter().map(|t| t.id.clone()).collect();
                let output_path = mix_output_path();
                let mut mixinfo = MixInfo {
                    status: "in_progress".into(),
                    total_tracks: tracks.len(),
                    completed_tracks: 1,
                    track_ids,
                    output_path: output_path.clone(),
                    updated_at: now_seconds(),
                    error: None,
                };
                write_mixinfo(&mixinfo);
                let mut writer = audio::MixWriter::create(&output_path)?;
                let mut tail = audio::decode_stereo(&tracks[0].path)?;
                let mut elapsed = Duration::from_secs(u64::from(tracks[0].duration));
                for pair in tracks.windows(2) {
                    let (from, to) = (&pair[0], &pair[1]);
                    elapsed = elapsed.saturating_add(Duration::from_secs(u64::from(to.duration)));
                    if elapsed > MIX_ALL_MAX_DURATION {
                        let error = format!(
                            "Mix all is limited to {} hours of combined audio per render.",
                            MIX_ALL_MAX_DURATION.as_secs() / 3600
                        );
                        mixinfo.status = "error".into();
                        mixinfo.error = Some(error.clone());
                        mixinfo.updated_at = now_seconds();
                        write_mixinfo(&mixinfo);
                        return Err(error);
                    }
                    let incoming = audio::decode_stereo(&to.path)?;
                    let plan = if !use_openrouter {
                        local_plan(from, to)
                    } else {
                        openrouter_plan(&key, from, to, 0)
                            .map(|(plan, _)| plan)
                            .unwrap_or_else(|_| local_plan(from, to))
                    };
                    tail = match audio::transition_stream(
                        &mut writer,
                        &tail,
                        &incoming,
                        from.bpm,
                        to.bpm,
                        &plan,
                    ) {
                        Ok(tail) => tail,
                        Err(error) => {
                            mixinfo.status = "error".into();
                            mixinfo.error = Some(error.clone());
                            mixinfo.updated_at = now_seconds();
                            write_mixinfo(&mixinfo);
                            return Err(error);
                        }
                    };
                    mixinfo.completed_tracks += 1;
                    mixinfo.updated_at = now_seconds();
                    write_mixinfo(&mixinfo);
                }
                writer.write_frames(&tail.samples)?;
                let duration = writer.finish()?;
                mixinfo.status = "done".into();
                mixinfo.updated_at = now_seconds();
                write_mixinfo(&mixinfo);
                Ok((output_path, duration))
            })();
            let _ = tx.send(WorkerMessage::PlaylistMixReady { first_id, result });
        });
    }

    /// Asks the user where to save the rendered mix as MP4 and kicks off the
    /// ffmpeg transcode on a background thread. The button that calls this
    /// only appears when `last_mix_path` is `Some`, so the guard is mostly
    /// defensive.
    fn export_mix_to_mp4(&mut self) {
        let Some(source) = self.last_mix_path.clone() else {
            self.notify(
                "No mix to export",
                "Create a mix first with Mix all or a playlist.",
                true,
            );
            return;
        };
        if !source.is_file() {
            self.notify(
                "Mix file missing",
                "The rendered mix is no longer on disk. Mix again to recreate it.",
                true,
            );
            return;
        }
        let timestamp = now_seconds();
        let default_name = format!("OpenMix-{timestamp}.mp4");
        let Some(mut output_path) = rfd::FileDialog::new()
            .set_title("Save mix as MP4")
            .set_file_name(&default_name)
            .add_filter("MP4 audio", &["mp4"])
            .save_file()
        else {
            return;
        };
        set_mp4_extension(&mut output_path);
        self.status = "Exporting mix to MP4…".into();
        let tx = self.tx.clone();
        let dest = output_path.clone();
        thread::spawn(move || {
            let result = export_wav_to_mp4(&source, &dest).map(|_| dest);
            let _ = tx.send(WorkerMessage::MixExportReady(result));
        });
    }

    fn openrouter_test(&mut self) {
        let key = self.key_input.trim().to_owned();
        if key.is_empty() {
            self.key_status = "Enter a key first.".into();
            return;
        }
        self.key_status = "Checking NVIDIA NIM".into();
        let tx = self.tx.clone();
        thread::spawn(move || {
            let result = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .map_err(|e| e.to_string())
                .and_then(|client| {
                    match send_completion(
                        &client,
                        &key,
                        vec![
                            json_message("system", "Reply with a short connection confirmation."),
                            json_message("user", "Confirm the OpenMix connection."),
                        ],
                        2048,
                        true,
                    ) {
                        Ok(_) => Ok(()),
                        Err(error) if error == "OpenRouter returned an empty completion." => Ok(()),
                        Err(error) => Err(error),
                    }
                });
            let _ = tx.send(WorkerMessage::OpenRouterTest(result));
        });
    }

    fn receive_messages(&mut self) {
        while let Ok(message) = self.rx.try_recv() {
            match message {
                WorkerMessage::Imported(Ok((tracks, problems))) => {
                    let mut added = 0;
                    let mut lookup = Vec::new();
                    for track in tracks {
                        if !self.data.tracks.iter().any(|old| old.id == track.id) {
                            lookup.push(track.clone());
                            self.data.tracks.push(track);
                            added += 1;
                        }
                    }
                    self.persist();
                    self.status = if problems.is_empty() {
                        format!("Added {added} track(s) to your library.")
                    } else {
                        format!(
                            "Added {added} track(s); skipped {} file(s). {}",
                            problems.len(),
                            problems.join(" · ")
                        )
                    };
                    if !lookup.is_empty() && self.data.use_musicbrainz {
                        self.lookup_metadata(lookup);
                    }
                }
                WorkerMessage::Imported(Err(error)) => self.status = error,
                WorkerMessage::MetadataFetched { track_id, result } => {
                    if let Ok(Some(metadata)) = result
                        && let Some(track) = self
                            .data
                            .tracks
                            .iter_mut()
                            .find(|track| track.id == track_id)
                    {
                        track.release_year = metadata.release_year;
                        if track.album == "Local library"
                            && let Some(album) = metadata.album
                        {
                            track.album = album;
                        }
                        track.musicbrainz_id = Some(metadata.musicbrainz_id);
                        if metadata.cover_art_path.is_some() {
                            track.cover_art_path = metadata.cover_art_path;
                        }
                        self.persist();
                    }
                }
                WorkerMessage::MetadataFetchFinished {
                    total,
                    matched,
                    misses,
                    unmatched,
                    errors,
                } => {
                    if let Some(first_error) = errors.first() {
                        let detail = if errors.len() > 1 {
                            format!("{first_error} (and {} more)", errors.len() - 1)
                        } else {
                            first_error.clone()
                        };
                        self.notify(
                            "MusicBrainz lookup failed",
                            format!("{detail} · {matched}/{total} matched"),
                            true,
                        );
                    } else if matched == 0 {
                        let first = unmatched
                            .first()
                            .map(|track| format!(" First: {track}."))
                            .unwrap_or_default();
                        self.notify(
                            "MusicBrainz lookup complete",
                            format!("No catalog match for {total} track(s).{first}"),
                            false,
                        );
                    } else {
                        self.notify(
                            "MusicBrainz lookup complete",
                            format!("Updated {matched}/{total} track(s); {misses} unmatched."),
                            false,
                        );
                    }
                }
                WorkerMessage::PlaylistMixReady { first_id, result } => match result {
                    Ok((path, duration)) => match self.audio.play_mix_file(&path, duration) {
                        Ok(()) => {
                            self.last_mix_path = Some(path.clone());
                            self.current = Some(first_id.clone());
                            self.queue.clear();
                            self.data.recent.retain(|id| id != &first_id);
                            self.data.recent.insert(0, first_id.clone());
                            self.data.recent.truncate(50);
                            if let Some(track) =
                                self.data.tracks.iter_mut().find(|t| t.id == first_id)
                            {
                                track.last_played = now_seconds();
                            }
                            self.persist();
                            self.status.clear();
                        }
                        Err(error) => self.status = error,
                    },
                    Err(error) => self.status = format!("Could not mix playlist: {error}"),
                },
                WorkerMessage::TransitionDecoded {
                    outgoing_id,
                    incoming_id,
                    result: Ok((outgoing, incoming, plan, rationale, local, from_bpm, to_bpm)),
                } => {
                    if self.current.as_deref().is_some_and(|id| id != outgoing_id) {
                        self.status = "Mix cancelled because the playing track changed.".into();
                        continue;
                    }
                    if !self.data.tracks.iter().any(|track| track.id == incoming_id) {
                        self.status =
                            "Mix cancelled because the incoming track was removed.".into();
                        continue;
                    }
                    let position = if self.current.as_deref() == Some(outgoing_id.as_str()) {
                        self.audio.position()
                    } else {
                        Duration::ZERO
                    };
                    let mut plan = plan;
                    if position > Duration::ZERO
                        && let (Some(from), Some(to)) = (
                            self.data
                                .tracks
                                .iter()
                                .find(|track| track.id == outgoing_id),
                            self.data
                                .tracks
                                .iter()
                                .find(|track| track.id == incoming_id),
                        )
                    {
                        let seconds = position.as_secs().min(u64::from(u32::MAX)) as u32;
                        if to.rise > 0 {
                            let cue = aligned_incoming_start(
                                from,
                                to,
                                seconds,
                                plan.entry_lead_seconds,
                                plan.fade_seconds,
                            );
                            plan.incoming_start_seconds = plan
                                .incoming_start_seconds
                                .clamp(cue.saturating_sub(8), cue.saturating_add(8));
                        }
                    }
                    let starting_position = position;
                    let tx = self.tx.clone();
                    thread::spawn(move || {
                        let tail =
                            audio::trim_start(&outgoing, position).unwrap_or(audio::StereoPcm {
                                samples: Vec::new(),
                                duration: Duration::ZERO,
                            });
                        let pcm = audio::transition(&tail, &incoming, from_bpm, to_bpm, &plan);
                        let export_path = mix_preview_output_path();
                        let export_path = write_pcm_to_wav(&pcm, &export_path)
                            .ok()
                            .map(|_| export_path);
                        let result = Ok((pcm, rationale, local, starting_position, export_path));
                        let _ = tx.send(WorkerMessage::TransitionReady {
                            outgoing_id,
                            incoming_id,
                            result,
                        });
                    });
                }
                WorkerMessage::TransitionDecoded {
                    result: Err(error), ..
                } => self.status = error,
                WorkerMessage::TransitionReady {
                    outgoing_id,
                    incoming_id,
                    result: Ok((pcm, rationale, local, starting_position, export_path)),
                } => {
                    if self.current.as_deref().is_some_and(|id| id != outgoing_id) {
                        self.status = "Mix cancelled because the playing track changed.".into();
                        continue;
                    }
                    if !self.data.tracks.iter().any(|track| track.id == incoming_id) {
                        self.status =
                            "Mix cancelled because the incoming track was removed.".into();
                        continue;
                    }
                    let position = self.audio.position().saturating_sub(starting_position);
                    match self.audio.play_pcm_from(pcm, position) {
                        Ok(()) => {
                            if let Some(export_path) = export_path {
                                self.last_mix_path = Some(export_path);
                            }
                            self.notes.insert(incoming_id.clone(), (rationale, local));
                            self.current = Some(incoming_id.clone());
                            self.data.recent.retain(|id| id != &incoming_id);
                            self.data.recent.insert(0, incoming_id.clone());
                            self.data.recent.truncate(50);
                            if let Some(track) = self
                                .data
                                .tracks
                                .iter_mut()
                                .find(|track| track.id == incoming_id)
                            {
                                track.last_played = now_seconds();
                            }
                            self.persist();
                            self.status.clear();
                        }
                        Err(error) => self.status = error,
                    }
                }
                WorkerMessage::TransitionReady {
                    result: Err(error), ..
                } => self.status = error,
                WorkerMessage::OpenRouterTest(Ok(())) => {
                    self.key_status = "Connected. The check sent no track information.".into()
                }
                WorkerMessage::OpenRouterTest(Err(error)) => {
                    self.key_status = format!("Connection failed: {error}")
                }
                WorkerMessage::MixExportReady(Ok(path)) => {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "mix.mp4".into());
                    self.notify("Export complete", format!("Saved {name}"), false);
                    self.status = format!("Exported {name}");
                }
                WorkerMessage::MixExportReady(Err(error)) => {
                    self.notify("Export failed", error.clone(), true);
                    self.status = error;
                }
            }
        }
    }

    fn update_shortcuts(&mut self, ctx: &egui::Context) {
        let (space, ctrl_k, ctrl_f, right, left, ctrl_s, ctrl_r) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::Space) && !i.modifiers.ctrl && !i.modifiers.command,
                i.key_pressed(egui::Key::K) && i.modifiers.command,
                i.key_pressed(egui::Key::F) && i.modifiers.command,
                i.key_pressed(egui::Key::ArrowRight) && i.modifiers.alt,
                i.key_pressed(egui::Key::ArrowLeft) && i.modifiers.alt,
                i.key_pressed(egui::Key::S) && i.modifiers.command,
                i.key_pressed(egui::Key::R) && i.modifiers.command,
            )
        });
        if space {
            self.toggle();
        }
        if ctrl_k || ctrl_f {
            self.search_focus = true;
            self.page = Page::Library;
        }
        if right {
            self.next();
        }
        if left {
            self.previous();
        }
        if ctrl_s {
            self.shuffle = !self.shuffle;
        }
        if ctrl_r {
            self.repeat = (self.repeat + 1) % 3;
        }
    }

    fn left_sidebar(&mut self, root: &mut egui::Ui) {
        egui::Panel::left("library_sidebar")
            .resizable(false)
            .default_size(232.0)
            .frame(
                Frame::new()
                    .fill(theme_sidebar())
                    .inner_margin(Margin::symmetric(14, 20)),
            )
            .show(root, |ui| {
                ui.horizontal(|ui| {
                    brand_mark(ui);
                    ui.label(
                        RichText::new("openmix")
                            .size(21.0)
                            .strong()
                            .color(theme_text()),
                    );
                });
                ui.add_space(28.0);
                self.nav_button(ui, Icon::Home, "Listen", Page::Listen);
                self.nav_button(ui, Icon::Mix, "Mixing", Page::Mixing);
                self.nav_button(ui, Icon::Library, "Your library", Page::Library);
                self.nav_button(ui, Icon::Liked, "Liked tracks", Page::Liked);
                ui.add_space(26.0);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("PLAYLISTS")
                            .size(10.0)
                            .strong()
                            .color(theme_quiet()),
                    );
                    if icon_button(ui, Icon::Add, "Create playlist").clicked() {
                        self.playlist_name.clear();
                        self.playlist_edit = None;
                        self.playlist_dialog = true;
                    }
                });
                ui.add_space(6.0);
                let playlists = self.data.playlists.clone();
                egui::ScrollArea::vertical()
                    .id_salt("playlist_nav")
                    .max_height(ui.available_height() - 100.0)
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                    .show(ui, |ui| {
                        for playlist in playlists {
                            let selected = self.active_playlist.as_deref()
                                == Some(playlist.id.as_str())
                                && self.page == Page::Playlist;
                            if nav_row(ui, Icon::Playlist, &playlist.name, selected).clicked() {
                                self.active_playlist = Some(playlist.id);
                                self.page = Page::Playlist;
                                self.sort = SortMode::PlaylistOrder;
                                self.sort_ascending = true;
                            }
                        }
                    });
                ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
                    if nav_row(ui, Icon::Settings, "Settings", self.page == Page::Settings)
                        .clicked()
                    {
                        self.page = Page::Settings;
                    }
                    ui.add_space(8.0);
                    ui.separator();
                    ui.label(
                        RichText::new(format!("{} local tracks", self.data.tracks.len()))
                            .size(11.0)
                            .color(theme_quiet()),
                    );
                    if !self.status.is_empty() {
                        ui.label(RichText::new(&self.status).size(11.0).color(theme_soft()));
                    }
                });
            });
    }

    fn nav_button(&mut self, ui: &mut egui::Ui, icon: Icon, label: &str, page: Page) {
        if nav_row(ui, icon, label, self.page == page).clicked() {
            self.page = page;
        }
    }

    fn right_now_panel(&mut self, root: &mut egui::Ui) {
        if !self.now_open {
            return;
        }
        egui::Panel::right("now_playing_panel")
            .default_size(292.0)
            .size_range(256.0..=380.0)
            .resizable(true)
            .frame(
                Frame::new()
                    .fill(theme_sidebar())
                    .inner_margin(Margin::symmetric(17, 20)),
            )
            .show(root, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("NOW PLAYING")
                            .size(10.0)
                            .strong()
                            .color(theme_quiet()),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if icon_button(ui, Icon::Close, "Hide this panel").clicked() {
                            self.now_open = false;
                        }
                    });
                });
                ui.add_space(18.0);
                if let Some(track) = self.current_track() {
                    record_art(ui, &track, ui.available_width().min(254.0));
                    ui.add_space(18.0);
                    ui.label(
                        RichText::new(&track.title)
                            .size(20.0)
                            .strong()
                            .color(theme_text()),
                    );
                    ui.label(RichText::new(&track.artist).size(14.0).color(theme_soft()));
                    ui.add_space(18.0);
                    ui.separator();
                    ui.add_space(12.0);
                    ui.label(
                        RichText::new("TRACK DETAILS")
                            .size(10.0)
                            .strong()
                            .color(theme_quiet()),
                    );
                    detail_row(ui, "Album", &track.album);
                    detail_row(
                        ui,
                        "Tempo",
                        &if track.bpm > 0 {
                            format!("{} BPM", track.bpm)
                        } else {
                            "Not analyzed".into()
                        },
                    );
                    detail_row(ui, "Length", &format_time(track.duration));
                    ui.add_space(16.0);
                    let like_label = if track.liked {
                        "In your liked tracks"
                    } else {
                        "Add to liked tracks"
                    };
                    if secondary_button(ui, like_label).clicked() {
                        self.toggle_like(&track.id);
                    }
                    if secondary_button(ui, "Add to queue").clicked() {
                        self.add_to_queue(&track.id);
                    }
                } else {
                    ui.add_space(80.0);
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            RichText::new("Choose a local track to see its details here.")
                                .size(13.0)
                                .color(theme_soft()),
                        );
                    });
                }
            });
    }

    fn player_bar(&mut self, root: &mut egui::Ui) {
        egui::Panel::bottom("player_bar")
            .default_size(96.0)
            .size_range(84.0..=140.0)
            .frame(
                Frame::new()
                    .fill(theme_surface())
                    .stroke(Stroke::new(1.0, theme_edge()))
                    .inner_margin(Margin::symmetric(16, 8)),
            )
            .show(root, |ui| {
                let track = self.current_track();
                let bar_rect = ui.available_rect_before_wrap();
                let full_h = bar_rect.height();
                let avail_w = bar_rect.width();

                // Fixed side widths; everything left over is the true center.
                let side_w = 220.0_f32.min(avail_w * 0.22).max(180.0);
                let center_w = (avail_w - side_w * 2.0).max(280.0);

                ui.horizontal(|ui| {
                    ui.set_min_height(full_h);
                    ui.spacing_mut().item_spacing.x = 0.0;

                    // ── LEFT: track info ──────────────────────────────────
                    ui.allocate_ui_with_layout(
                        egui::vec2(side_w, full_h),
                        Layout::left_to_right(Align::Center),
                        |ui| {
                            ui.spacing_mut().item_spacing.x = 10.0;
                            if let Some(t) = &track {
                                record_art(ui, t, 42.0);
                            }
                            ui.vertical(|ui| {
                                if let Some(t) = &track {
                                    ui.label(
                                        RichText::new(&t.title)
                                            .size(13.0)
                                            .strong()
                                            .color(theme_text()),
                                    );
                                    ui.label(
                                        RichText::new(&t.artist).size(11.0).color(theme_soft()),
                                    );
                                } else {
                                    ui.label(
                                        RichText::new("Nothing playing")
                                            .size(13.0)
                                            .strong()
                                            .color(theme_text()),
                                    );
                                    ui.label(
                                        RichText::new("Your local music, on your terms")
                                            .size(11.0)
                                            .color(theme_soft()),
                                    );
                                }
                            });
                        },
                    );

                    // ── CENTER: transport + timeline ──────────────────────
                    // Use a vertical layout with Align::Center so every child
                    // is horizontally centered inside the column.
                    ui.allocate_ui_with_layout(
                        egui::vec2(center_w, full_h),
                        Layout::top_down(Align::Center),
                        |ui| {
                            ui.set_width(center_w);
                            ui.set_height(full_h);
                            ui.spacing_mut().item_spacing.y = 0.0;

                            const CONTROLS_HEIGHT: f32 = 40.0;
                            const CONTROLS_WIDTH: f32 = 204.0;
                            const TIMELINE_HEIGHT: f32 = 28.0;
                            const TIMELINE_GAP: f32 = 6.0;
                            const TIME_LABEL_WIDTH: f32 = 42.0;
                            const TIMELINE_ITEM_SPACING: f32 = 8.0;

                            let center_rect = ui.max_rect();
                            let controls_center_y =
                                center_rect.center().y - (TIMELINE_HEIGHT + TIMELINE_GAP) * 0.5;
                            let controls_rect = egui::Rect::from_center_size(
                                egui::pos2(center_rect.center().x, controls_center_y),
                                egui::vec2(CONTROLS_WIDTH, CONTROLS_HEIGHT),
                            );

                            // Keep the transport controls centered independently
                            // of the metadata and utility panels on either side.
                            let mut controls_ui = ui.new_child(
                                egui::UiBuilder::new()
                                    .id_salt("player_transport_controls")
                                    .max_rect(controls_rect)
                                    .layout(Layout::left_to_right(Align::Center)),
                            );
                            controls_ui.spacing_mut().item_spacing.x = 6.0;
                            if toggle_button(
                                &mut controls_ui,
                                Icon::Shuffle,
                                self.shuffle,
                                "Shuffle",
                            )
                            .clicked()
                            {
                                self.shuffle = !self.shuffle;
                            }
                            if icon_button(&mut controls_ui, Icon::Previous, "Previous track")
                                .clicked()
                            {
                                self.previous();
                            }
                            let playing = self.audio.is_playing();
                            if transport_button(
                                &mut controls_ui,
                                if playing { Icon::Pause } else { Icon::Play },
                            )
                            .clicked()
                            {
                                self.toggle();
                            }
                            if icon_button(&mut controls_ui, Icon::Next, "Next track").clicked() {
                                self.next();
                            }
                            if toggle_button(
                                &mut controls_ui,
                                if self.repeat == 1 {
                                    Icon::RepeatOne
                                } else {
                                    Icon::Repeat
                                },
                                self.repeat > 0,
                                "Repeat",
                            )
                            .clicked()
                            {
                                self.repeat = (self.repeat + 1) % 3;
                            }

                            let pos = self.audio.position().as_secs_f32();
                            let len = self.audio.duration.as_secs_f32().max(1.0);
                            let timeline_top = controls_rect.bottom() + TIMELINE_GAP;
                            let timeline_rect = egui::Rect::from_min_max(
                                egui::pos2(center_rect.left(), timeline_top),
                                egui::pos2(center_rect.right(), timeline_top + TIMELINE_HEIGHT),
                            );

                            let elapsed_rect = egui::Rect::from_min_max(
                                timeline_rect.left_top(),
                                egui::pos2(
                                    timeline_rect.left() + TIME_LABEL_WIDTH,
                                    timeline_rect.bottom(),
                                ),
                            );
                            let slider_rect = egui::Rect::from_min_max(
                                egui::pos2(
                                    elapsed_rect.right() + TIMELINE_ITEM_SPACING,
                                    timeline_rect.top(),
                                ),
                                egui::pos2(
                                    timeline_rect.right()
                                        - TIME_LABEL_WIDTH
                                        - TIMELINE_ITEM_SPACING,
                                    timeline_rect.bottom(),
                                ),
                            );
                            let duration_rect = egui::Rect::from_min_max(
                                egui::pos2(
                                    timeline_rect.right() - TIME_LABEL_WIDTH,
                                    timeline_rect.top(),
                                ),
                                timeline_rect.right_bottom(),
                            );

                            let mut elapsed_ui = ui.new_child(
                                egui::UiBuilder::new()
                                    .id_salt("player_elapsed_time")
                                    .max_rect(elapsed_rect)
                                    .layout(Layout::left_to_right(Align::Center)),
                            );
                            elapsed_ui.label(
                                RichText::new(format_time(pos as u32))
                                    .size(10.0)
                                    .color(theme_quiet()),
                            );

                            let mut slider_ui = ui.new_child(
                                egui::UiBuilder::new()
                                    .id_salt("player_timeline_slider")
                                    .max_rect(slider_rect)
                                    .layout(Layout::left_to_right(Align::Center)),
                            );
                            slider_ui.spacing_mut().slider_width = slider_rect.width();
                            slider_ui.spacing_mut().interact_size.y = TIMELINE_HEIGHT;
                            let mut value = pos.min(len);
                            let response = slider_ui.add_sized(
                                [slider_rect.width(), TIMELINE_HEIGHT],
                                egui::Slider::new(&mut value, 0.0..=len)
                                    .show_value(false)
                                    .handle_shape(egui::style::HandleShape::Circle)
                                    .trailing_fill(true),
                            );
                            if (response.drag_stopped()
                                || (response.clicked() && response.hovered()))
                                && let Err(error) = self.audio.seek(Duration::from_secs_f32(value))
                            {
                                self.status = error;
                            }

                            let mut duration_ui = ui.new_child(
                                egui::UiBuilder::new()
                                    .id_salt("player_duration_time")
                                    .max_rect(duration_rect)
                                    .layout(Layout::right_to_left(Align::Center)),
                            );
                            duration_ui.label(
                                RichText::new(format_time(len as u32))
                                    .size(10.0)
                                    .color(theme_quiet()),
                            );
                        },
                    );

                    // ── RIGHT: volume / queue / now-playing / export ──────
                    ui.allocate_ui_with_layout(
                        egui::vec2(side_w, full_h),
                        Layout::right_to_left(Align::Center),
                        |ui| {
                            ui.spacing_mut().item_spacing.x = 6.0;
                            let mut volume = self.audio.volume;
                            let muted = self.muted;
                            if let Some(action) = volume_control(ui, &mut volume, muted) {
                                match action {
                                    VolumeAction::Set(value) => {
                                        self.audio.set_volume(value);
                                        self.muted = value <= 0.001;
                                        if !self.muted {
                                            self.volume_before_mute = value;
                                        }
                                    }
                                    VolumeAction::ToggleMute => {
                                        if self.audio.volume > 0.001 {
                                            self.volume_before_mute = self.audio.volume;
                                            self.audio.set_volume(0.0);
                                            self.muted = true;
                                        } else {
                                            let restored = if self.volume_before_mute > 0.001 {
                                                self.volume_before_mute
                                            } else {
                                                0.8
                                            };
                                            self.audio.set_volume(restored);
                                            self.muted = false;
                                        }
                                    }
                                }
                            }
                            if icon_button(ui, Icon::Queue, "Open play queue").clicked() {
                                self.page = Page::Queue;
                            }
                            if icon_button(ui, Icon::NowPlaying, "Toggle Now Playing panel")
                                .clicked()
                            {
                                self.now_open = !self.now_open;
                            }
                            if self.last_mix_path.is_some() {
                                if export_button(ui, "Export finished mix as MP4").clicked() {
                                    self.export_mix_to_mp4();
                                }
                            }
                        },
                    );
                });

                if let Some(error) = self.audio.error.as_deref() {
                    ui.colored_label(theme_red(), error);
                }
            });
    }

    fn content(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_rect_before_wrap();
        ui.painter()
            .rect_filled(available, CornerRadius::ZERO, theme_bg());
        Frame::new()
            .fill(theme_bg())
            .inner_margin(Margin::symmetric(28, 22))
            .show(ui, |ui| {
                ui.set_min_size(Vec2::new(
                    (available.width() - 56.0).max(0.0),
                    (available.height() - 44.0).max(0.0),
                ));
                self.topbar(ui);
                ui.add_space(18.0);
                if self.animated_page != self.page {
                    self.animated_page = self.page;
                    self.page_animation_target = !self.page_animation_target;
                }
                let enter = ui.ctx().animate_bool_responsive(
                    ui.make_persistent_id("page-enter"),
                    self.page_animation_target,
                );
                let offset = if self.page_animation_target {
                    1.0 - enter
                } else {
                    enter
                };
                ui.add_space(offset * 6.0);
                egui::ScrollArea::vertical()
                    .id_salt("main_page_scroll")
                    .auto_shrink([false, false])
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                    .show(ui, |ui| match self.page {
                        Page::Listen => self.listen_page(ui),
                        Page::Mixing => self.mixing_page(ui),
                        Page::Library | Page::Liked | Page::Playlist => self.library_page(ui),
                        Page::Queue => self.queue_page(ui),
                        Page::Settings => self.settings_page(ui),
                    });
            });
    }

    fn topbar(&mut self, ui: &mut egui::Ui) {
        let title = match self.page {
            Page::Listen => "Listen".to_owned(),
            Page::Mixing => "Mixing".to_owned(),
            Page::Library => "Your library".to_owned(),
            Page::Liked => "Liked tracks".to_owned(),
            Page::Playlist => self
                .active_playlist
                .as_ref()
                .and_then(|id| self.data.playlists.iter().find(|p| &p.id == id))
                .map(|p| p.name.clone())
                .unwrap_or_else(|| "Playlist".to_owned()),
            Page::Queue => "Play queue".to_owned(),
            Page::Settings => "Settings".to_owned(),
        };
        let is_collection = matches!(self.page, Page::Library | Page::Liked | Page::Playlist);
        let can_search = is_collection || matches!(self.page, Page::Listen | Page::Mixing);
        let hint = if is_collection {
            "Filter this collection"
        } else {
            "Search your library"
        };
        let compact = ui.available_width() < 880.0;
        ui.horizontal(|ui| {
            ui.label(RichText::new(title).size(29.0).strong().color(theme_text()));
            if can_search && !compact {
                ui.add_space(12.0);
                let width = if is_collection { 235.0 } else { 250.0 };
                self.search_box(ui, hint, width);
                if !is_collection
                    && !self.search.trim().is_empty()
                    && small_secondary_button(ui, "Show results").clicked()
                {
                    self.page = Page::Library;
                }
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if primary_button(ui, "Import files").clicked() {
                    self.import_files();
                }
            });
        });
        if can_search && compact {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let show_results = !is_collection && !self.search.trim().is_empty();
                let width = ui.available_width() - if show_results { 104.0 } else { 0.0 };
                self.search_box(ui, hint, width);
                if show_results && small_secondary_button(ui, "Show results").clicked() {
                    self.page = Page::Library;
                }
            });
        }
        ui.add_space(7.0);
        ui.separator();
    }

    fn search_box(&mut self, ui: &mut egui::Ui, hint: &str, width: f32) {
        let response = ui
            .horizontal(|ui| {
                inline_icon(ui, Icon::Search);
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text(hint)
                    .desired_width((width - 32.0).max(120.0))
                    .show(ui)
                    .response
            })
            .inner;
        if self.search_focus {
            response.request_focus();
            self.search_focus = false;
        }
    }

    fn listen_page(&mut self, ui: &mut egui::Ui) {
        ui.add_space(12.0);
        ui.label(
            RichText::new("Your music, in one place.")
                .size(15.0)
                .color(theme_soft()),
        );
        ui.add_space(24.0);
        let mut added_tracks = self.data.tracks.clone();
        added_tracks.sort_by_key(|track| std::cmp::Reverse(track.added));
        if !added_tracks.is_empty() {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Recently added")
                        .size(18.0)
                        .strong()
                        .color(theme_text()),
                );
                if added_tracks.len() >= 2 && primary_button(ui, "Mix all").clicked() {
                    self.mix_all(added_tracks.clone());
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.link("View library").clicked() {
                        self.page = Page::Library;
                    }
                });
            });
            ui.add_space(8.0);
            for track in added_tracks.into_iter().take(8) {
                self.track_row(ui, &track, false);
            }
            ui.add_space(24.0);
        }
        let recent_ids = self.data.recent.clone();
        let recent: Vec<Track> = recent_ids
            .iter()
            .filter_map(|id| self.data.tracks.iter().find(|t| &t.id == id).cloned())
            .collect();
        if recent.is_empty() && self.data.tracks.is_empty() {
            self.empty_state(
                ui,
                "No music here yet",
                "Import local audio files to start building your own library.",
            );
        } else if !recent.is_empty() {
            self.section_header(ui, "Recently played", Some(("View library", Page::Library)));
            ui.add_space(8.0);
            for track in recent.into_iter().take(8) {
                self.track_row(ui, &track, false);
            }
        }
        ui.add_space(30.0);
        if !self.data.tracks.is_empty() {
            self.section_header(ui, "Ready to mix", Some(("Open Mixing", Page::Mixing)));
            ui.label(
                RichText::new(
                    "Choose a starting track and OpenMix will find compatible local files.",
                )
                .size(12.0)
                .color(theme_soft()),
            );
        }
    }

    fn mixing_page(&mut self, ui: &mut egui::Ui) {
        ui.label(
            RichText::new("Create a hand-off from the music you own.")
                .size(14.0)
                .color(theme_soft()),
        );
        ui.add_space(22.0);
        if self.data.tracks.len() < 2 {
            self.empty_state(
                ui,
                "Mixing needs two tracks",
                "Import at least two local MP3 or WAV files. Analysis stays on this device.",
            );
            return;
        }
        let selected = self
            .current_track()
            .or_else(|| self.data.tracks.first().cloned());
        let Some(outgoing) = selected else {
            return;
        };
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("FROM")
                    .size(10.0)
                    .strong()
                    .color(theme_quiet()),
            );
            ui.add_space(8.0);
            ui.label(
                RichText::new(&outgoing.title)
                    .size(16.0)
                    .strong()
                    .color(theme_text()),
            );
            ui.label(
                RichText::new(format!(
                    "{}  ·  {} BPM",
                    outgoing.artist,
                    bpm_label(outgoing.bpm)
                ))
                .size(12.0)
                .color(theme_soft()),
            );
            if self.current.is_none() && secondary_button(ui, "Choose from library").clicked() {
                self.page = Page::Library;
            }
        });
        ui.add_space(22.0);
        self.section_header(ui, "Compatible next tracks", None);
        ui.add_space(8.0);
        let mut candidates = self.data.tracks.clone();
        candidates.retain(|t| t.id != outgoing.id);
        candidates.sort_by(|a, b| mix_score(&outgoing, a).total_cmp(&mix_score(&outgoing, b)));
        for track in candidates.into_iter().take(12) {
            self.suggestion_row(&outgoing, &track, ui);
        }
        ui.add_space(12.0);
        ui.label(
            RichText::new(
                "OpenMix compares tempo, full-track energy changes, and release details to find compatible hand-offs.",
            )
            .size(11.0)
            .color(theme_quiet()),
        );
    }

    fn suggestion_row(&mut self, from: &Track, to: &Track, ui: &mut egui::Ui) {
        let reason = if from.bpm > 0 && to.bpm > 0 {
            format!(
                "{} BPM  ·  {}",
                bpm_label(to.bpm),
                if (from.bpm as i32 - to.bpm as i32).abs() <= 5 {
                    "close tempo"
                } else {
                    "half-time compatible"
                }
            )
        } else {
            format!("{} BPM  ·  tempo not detected", bpm_label(to.bpm))
        };
        Frame::new()
            .fill(theme_surface())
            .inner_margin(Margin::symmetric(12, 10))
            .stroke(Stroke::new(1.0, theme_edge()))
            .corner_radius(CornerRadius::same(7))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    record_art(ui, to, 42.0);
                    ui.vertical(|ui| {
                        ui.label(
                            RichText::new(&to.title)
                                .size(14.0)
                                .strong()
                                .color(theme_text()),
                        );
                        ui.label(
                            RichText::new(format!("{}  ·  {reason}", to.artist))
                                .size(11.0)
                                .color(theme_soft()),
                        );
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if primary_button(ui, "Mix now").clicked() {
                            self.queue_transition(from.clone(), to.clone());
                        }
                        if secondary_button(ui, "Play next").clicked() {
                            self.add_to_queue(&to.id);
                        }
                    });
                });
                if let Some((note, fallback)) = self.notes.get(&to.id) {
                    ui.add_space(7.0);
                    ui.label(
                        RichText::new(if *fallback {
                            format!("LOCAL PLAN  ·  {note}")
                        } else {
                            format!("AI PLAN  ·  {note}")
                        })
                        .size(12.0)
                        .color(if *fallback {
                            theme_soft()
                        } else {
                            theme_text()
                        }),
                    );
                }
            });
        ui.add_space(7.0);
    }

    fn library_page(&mut self, ui: &mut egui::Ui) {
        let tracks = self.visible_tracks();
        let heading = match self.page {
            Page::Liked => format!("{} liked tracks", tracks.len()),
            Page::Playlist => format!("{} tracks", tracks.len()),
            _ => format!("{} tracks", tracks.len()),
        };
        ui.horizontal(|ui| {
            ui.label(RichText::new(heading).size(12.0).color(theme_soft()));
            ui.add_space(16.0);
            egui::ComboBox::from_id_salt("sort_mode")
                .selected_text(format!("Sort: {}", self.sort.label()))
                .show_ui(ui, |ui| {
                    for mode in [
                        SortMode::RecentlyAdded,
                        SortMode::PlaylistOrder,
                        SortMode::Title,
                        SortMode::Artist,
                        SortMode::Duration,
                        SortMode::Bpm,
                    ] {
                        ui.selectable_value(&mut self.sort, mode, mode.label());
                    }
                });
            if icon_button(
                ui,
                if self.sort_ascending {
                    Icon::SortUp
                } else {
                    Icon::SortDown
                },
                "Reverse sort",
            )
            .clicked()
            {
                self.sort_ascending = !self.sort_ascending;
            }
            if !tracks.is_empty()
                && self.selected_tracks.len() < tracks.len()
                && small_secondary_button(ui, "Select all").clicked()
            {
                let ids: HashSet<String> = tracks.iter().map(|t| t.id.clone()).collect();
                self.selected_tracks.extend(ids);
            }
            if self.page == Page::Playlist {
                if !tracks.is_empty() && primary_button(ui, "Play playlist").clicked() {
                    self.queue = tracks.iter().skip(1).map(|t| t.id.clone()).collect();
                    self.play_id(&tracks[0].id);
                }
                if tracks.len() >= 2 && primary_button(ui, "Mix all").clicked() {
                    self.mix_all(tracks.clone());
                }
                let mut rename = false;
                let mut delete = false;
                menu_button(ui, "More", |ui| {
                    rename = secondary_button(ui, "Rename playlist").clicked();
                    delete = danger_button(ui, "Delete playlist").clicked();
                });
                if rename {
                    self.playlist_edit = self.active_playlist.clone();
                    self.playlist_name = self
                        .active_playlist
                        .as_ref()
                        .and_then(|id| self.data.playlists.iter().find(|p| &p.id == id))
                        .map(|p| p.name.clone())
                        .unwrap_or_default();
                    self.playlist_dialog = true;
                }
                if delete {
                    self.playlist_delete = self.active_playlist.clone();
                }
            }
        });
        if !self.selected_tracks.is_empty() {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("{} selected", self.selected_tracks.len()))
                        .color(theme_red()),
                );
                ui.add_space(12.0);
                let mut bulk_action: Option<BulkPlaylistAction> = None;
                menu_button(ui, "Add to playlist", |ui| {
                    if secondary_button(ui, "New playlist…").clicked() {
                        bulk_action = Some(BulkPlaylistAction::New);
                        ui.close();
                    }
                    if !self.data.playlists.is_empty() {
                        ui.separator();
                        let playlists = self.data.playlists.clone();
                        for p in playlists {
                            if secondary_button(ui, p.name.as_str()).clicked() {
                                bulk_action = Some(BulkPlaylistAction::Existing(p.id));
                                ui.close();
                            }
                        }
                    }
                });
                match bulk_action {
                    Some(BulkPlaylistAction::New) => {
                        self.playlist_dialog = true;
                        self.playlist_edit = None;
                        self.playlist_name = String::new();
                    }
                    Some(BulkPlaylistAction::Existing(id)) => {
                        if let Some(p) = self.data.playlists.iter_mut().find(|p| p.id == id) {
                            for track_id in &self.selected_tracks {
                                if !p.tracks.contains(track_id) {
                                    p.tracks.push(track_id.clone());
                                }
                            }
                            self.persist();
                            self.selected_tracks.clear();
                        }
                    }
                    None => {}
                }
                if ghost_button(ui, "Clear selection").clicked() {
                    self.selected_tracks.clear();
                }
            });
            ui.separator();
        }
        ui.add_space(12.0);
        if tracks.is_empty() {
            let empty_playlist = self.page == Page::Playlist;
            let heading = if !self.search.is_empty() {
                "No results"
            } else if empty_playlist {
                "This playlist is empty"
            } else {
                "Your library is empty"
            };
            let detail = if !self.search.is_empty() {
                "Try another search or clear the filter."
            } else if empty_playlist {
                "Add tracks from the library with each track’s menu."
            } else {
                "Choose the files you want to listen to. OpenMix does not bundle music."
            };
            self.empty_state(ui, heading, detail);
            return;
        }
        ui.horizontal(|ui| {
            ui.add_space(105.0);
            ui.label(
                RichText::new("TITLE")
                    .size(10.0)
                    .strong()
                    .color(theme_quiet()),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    RichText::new("LENGTH")
                        .size(10.0)
                        .strong()
                        .color(theme_quiet()),
                );
                ui.add_space(62.0);
                ui.label(
                    RichText::new("BPM")
                        .size(10.0)
                        .strong()
                        .color(theme_quiet()),
                );
            });
        });
        ui.separator();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
            .show_rows(ui, 58.0, tracks.len(), |ui, range| {
                for index in range {
                    self.track_row(ui, &tracks[index], true);
                }
            });
    }

    fn track_row(&mut self, ui: &mut egui::Ui, track: &Track, show_actions: bool) {
        let selected = self.current.as_deref() == Some(track.id.as_str());
        Frame::new()
            .fill(if selected {
                theme_surface_2()
            } else {
                Color32::TRANSPARENT
            })
            .inner_margin(Margin::symmetric(7, 5))
            .corner_radius(CornerRadius::same(5))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let play = icon_button(
                        ui,
                        if selected && self.audio.is_playing() {
                            Icon::Pause
                        } else {
                            Icon::Play
                        },
                        if selected && self.audio.is_playing() {
                            "Pause"
                        } else {
                            "Play"
                        },
                    );
                    if play.clicked() {
                        if selected && !self.audio.ended() {
                            self.toggle();
                        } else {
                            self.play_id(&track.id);
                        }
                    }
                    ui.add_space(4.0);
                    let mut selected_val = self.selected_tracks.contains(&track.id);
                    if ui.checkbox(&mut selected_val, "").changed() {
                        if selected_val {
                            self.selected_tracks.insert(track.id.clone());
                        } else {
                            self.selected_tracks.remove(&track.id);
                        }
                    }
                    ui.add_space(4.0);
                    record_art(ui, track, 36.0);
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            let title = ui.selectable_label(
                                false,
                                RichText::new(&track.title).size(13.0).color(if selected {
                                    theme_red()
                                } else {
                                    theme_text()
                                }),
                            );
                            title.context_menu(|ui| self.track_context_menu(ui, track));
                        });
                        let mut details = vec![track.artist.clone()];
                        if let Some(year) = &track.release_year {
                            details.push(year.clone());
                        }
                        if track.album != "Local library" {
                            details.push(track.album.clone());
                        }
                        ui.label(
                            RichText::new(details.join("  ·  "))
                                .size(11.0)
                                .color(theme_soft()),
                        );
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(
                            RichText::new(format_time(track.duration))
                                .size(11.0)
                                .color(theme_soft()),
                        );
                        ui.add_space(18.0);
                        ui.label(
                            RichText::new(bpm_label(track.bpm))
                                .size(11.0)
                                .color(theme_soft()),
                        );
                        if show_actions {
                            ui.add_space(12.0);
                            menu_button(
                                ui,
                                RichText::new("⋯").size(18.0).color(theme_soft()),
                                |ui| self.track_context_menu(ui, track),
                            )
                            .response
                            .on_hover_text("Track actions");
                        }
                    });
                });
            });
    }

    fn track_context_menu(&mut self, ui: &mut egui::Ui, track: &Track) {
        if primary_button(ui, "Play now").clicked() {
            self.play_id(&track.id);
            ui.close();
        }
        if secondary_button(ui, "Add to queue").clicked() {
            self.add_to_queue(&track.id);
            ui.close();
        }
        let like_label = if track.liked {
            "Remove from liked tracks"
        } else {
            "Add to liked tracks"
        };
        if secondary_button(ui, like_label).clicked() {
            self.toggle_like(&track.id);
            ui.close();
        }
        ui.separator();
        if danger_button(ui, "Remove from OpenMix")
            .on_hover_text(
                "Removes it from your library and playlists. The audio file stays on disk.",
            )
            .clicked()
        {
            self.remove_track(&track.id);
            ui.close();
        }
        if self.page == Page::Playlist {
            if danger_button(ui, "Remove from this playlist").clicked() {
                if let Some(playlist) = self
                    .active_playlist
                    .as_ref()
                    .and_then(|id| self.data.playlists.iter_mut().find(|p| &p.id == id))
                {
                    playlist.tracks.retain(|id| id != &track.id);
                    self.persist();
                }
                ui.close();
            }
            let playlist_id = self.active_playlist.clone();
            let position = playlist_id.as_ref().and_then(|id| {
                self.data
                    .playlists
                    .iter()
                    .find(|playlist| &playlist.id == id)
                    .and_then(|playlist| {
                        playlist
                            .tracks
                            .iter()
                            .position(|item| item == &track.id)
                            .map(|index| (index, playlist.tracks.len()))
                    })
            });
            let move_earlier = position.is_some_and(|(index, _)| index > 0)
                && secondary_button(ui, "Move earlier in playlist").clicked();
            let move_later = position.is_some_and(|(index, len)| index + 1 < len)
                && secondary_button(ui, "Move later in playlist").clicked();
            if let (Some(id), Some((index, _))) = (playlist_id, position)
                && (move_earlier || move_later)
            {
                if let Some(playlist) = self.data.playlists.iter_mut().find(|p| p.id == id) {
                    let target = if move_earlier { index - 1 } else { index + 1 };
                    playlist.tracks.swap(index, target);
                    self.persist();
                }
                ui.close();
            }
        }
        if !self.data.playlists.is_empty() {
            ui.separator();
            menu_button(ui, "Add to playlist", |ui| {
                for playlist in self.data.playlists.clone() {
                    if secondary_button(ui, playlist.name.as_str()).clicked() {
                        if let Some(p) =
                            self.data.playlists.iter_mut().find(|p| p.id == playlist.id)
                            && !p.tracks.contains(&track.id)
                        {
                            p.tracks.push(track.id.clone());
                            self.persist();
                        }
                        ui.close();
                    }
                }
            });
        }
    }

    fn remove_track(&mut self, id: &str) {
        let Some(track) = self.data.tracks.iter().find(|track| track.id == id) else {
            return;
        };
        let title = track.title.clone();
        if self.current.as_deref() == Some(id) {
            if let Some(player) = self.audio.player.take() {
                player.stop();
            }
            self.audio.duration = Duration::ZERO;
            self.current = None;
        }
        self.data.tracks.retain(|track| track.id != id);
        self.data.recent.retain(|recent| recent != id);
        self.queue.retain(|queued| queued != id);
        for playlist in &mut self.data.playlists {
            playlist.tracks.retain(|track_id| track_id != id);
        }
        self.notes.remove(id);
        self.persist();
        self.status = format!("Removed {title} from OpenMix. The local file was kept.");
    }

    fn queue_page(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("{} tracks queued", self.queue.len()))
                    .size(12.0)
                    .color(theme_soft()),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if !self.queue.is_empty() && danger_button(ui, "Clear queue").clicked() {
                    self.queue.clear();
                }
                if !self.queue.is_empty() && secondary_button(ui, "Shuffle queue").clicked() {
                    shuffle_slice(&mut self.queue);
                }
                if !self.queue.is_empty() && primary_button(ui, "Play queue").clicked() {
                    self.next();
                }
            });
        });
        ui.add_space(14.0);
        if let Some(current) = self.current_track() {
            self.section_header(ui, "NOW PLAYING", None);
            self.track_row(ui, &current, false);
        }
        self.section_header(ui, "UP NEXT", None);
        ui.add_space(6.0);
        if self.queue.is_empty() {
            ui.label(
                RichText::new("Nothing queued. Add tracks from your library or Mixing.")
                    .size(13.0)
                    .color(theme_soft()),
            );
            return;
        }
        let mut remove_index = None;
        let mut move_action = None;
        let queue = self.queue.clone();
        for (index, id) in queue.iter().enumerate() {
            let Some(track) = self.data.tracks.iter().find(|t| &t.id == id).cloned() else {
                continue;
            };
            Frame::new()
                .fill(theme_surface())
                .inner_margin(Margin::symmetric(10, 7))
                .corner_radius(CornerRadius::same(5))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!("{:02}", index + 1))
                                .size(11.0)
                                .color(theme_quiet()),
                        );
                        record_art(ui, &track, 36.0);
                        ui.vertical(|ui| {
                            ui.label(RichText::new(&track.title).size(13.0).color(theme_text()));
                            ui.label(RichText::new(&track.artist).size(11.0).color(theme_soft()));
                        });
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if icon_button(ui, Icon::Remove, "Remove from queue").clicked() {
                                remove_index = Some(index);
                            }
                            if index + 1 < self.queue.len()
                                && icon_button(ui, Icon::MoveDown, "Move down").clicked()
                            {
                                move_action = Some((index, index + 1));
                            }
                            if index > 0 && icon_button(ui, Icon::MoveUp, "Move up").clicked() {
                                move_action = Some((index, index - 1));
                            }
                        });
                    });
                });
            ui.add_space(3.0);
        }
        if let Some(index) = remove_index {
            self.queue.remove(index);
        }
        if let Some((from, to)) = move_action {
            self.queue.swap(from, to);
        }
    }

    fn settings_page(&mut self, ui: &mut egui::Ui) {
        ui.label(
            RichText::new("Appearance")
                .size(17.0)
                .strong()
                .color(theme_text()),
        );
        ui.add_space(5.0);
        ui.label(
            RichText::new(
                "Choose a dark palette built around OpenMix's red outlines. Your selection is saved automatically.",
            )
            .size(12.0)
            .color(theme_soft()),
        );
        ui.add_space(12.0);
        let active_theme = self.data.theme;
        let mut chosen_theme = None;
        ui.columns(2, |columns| {
            for (column_index, column) in columns.iter_mut().enumerate() {
                column.vertical(|ui| {
                    for theme in AppTheme::ALL.iter().skip(column_index).step_by(2) {
                        if theme_card(ui, *theme, active_theme == *theme) {
                            chosen_theme = Some(*theme);
                        }
                        ui.add_space(10.0);
                    }
                });
            }
        });
        if let Some(theme) = chosen_theme {
            self.change_theme(ui.ctx(), theme);
        }
        ui.add_space(14.0);
        ui.separator();
        ui.add_space(18.0);
        ui.label(
            RichText::new("NVIDIA NIM")
                .size(17.0)
                .strong()
                .color(theme_text()),
        );
        ui.add_space(5.0);
        ui.label(
            RichText::new(
                "AI mix planning is optional. When enabled with a key, OpenMix sends track metadata and local audio-analysis estimates a NVIDIA NIM; no audio, lyrics, or file paths are sent.",
            )
            .size(12.0)
            .color(theme_soft()),
        );
        let mut ai_enabled = self.data.use_openrouter;
        if ui
            .checkbox(&mut ai_enabled, "Use NVIDIA NIM for mix planning")
            .changed()
        {
            self.data.use_openrouter = ai_enabled;
            self.persist();
        }
        ui.label(
            RichText::new("Turn this off to use the local mix planner, even when a key is saved.")
                .size(11.0)
                .color(theme_quiet()),
        );
        ui.add_space(14.0);
        ui.label(RichText::new(MODEL).size(11.0).color(theme_quiet()));
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.key_input)
                    .password(!self.key_visible)
                    .hint_text("Paste your NVIDIA API key")
                    .desired_width(360.0),
            );
            let visibility_label = if self.key_visible { "Hide" } else { "Show" };
            if secondary_button(ui, visibility_label).clicked() {
                self.key_visible = !self.key_visible;
            }
        });
        ui.label(RichText::new("Ctrl+V to paste. Your key is never sent to the model provider except in the authorization header.").size(11.0).color(theme_quiet()));
        ui.horizontal(|ui| {
            if primary_button(ui, "Save key").clicked() {
                self.key_input = self.key_input.trim().to_owned();
                match write_secret(&self.key_input) {
                    Ok(message) => self.key_status = message,
                    Err(error) => self.key_status = error,
                }
            }
            if ui
                .add_enabled_ui(self.data.use_openrouter, |ui| {
                    primary_button(ui, "Test connection")
                })
                .inner
                .clicked()
            {
                self.openrouter_test();
            }
            if danger_button(ui, "Clear key").clicked() {
                self.key_input.clear();
                self.key_status = match clear_secret() {
                    Ok(()) => "Key cleared from this session and the credential store.".into(),
                    Err(error) => {
                        format!("Key cleared for this session; credential store: {error}")
                    }
                };
            }
        });
        ui.label(
            RichText::new(&self.key_status)
                .size(11.0)
                .color(theme_soft()),
        );
        ui.add_space(22.0);
        ui.separator();
        ui.add_space(12.0);
        ui.label(
            RichText::new("MusicBrainz")
                .size(17.0)
                .strong()
                .color(theme_text()),
        );
        ui.label(
            RichText::new(
                "Metadata lookup is optional and works offline when disabled. Public MusicBrainz lookups need no API key.",
            )
            .size(12.0)
            .color(theme_soft()),
        );
        let mut metadata_enabled = self.data.use_musicbrainz;
        if ui
            .checkbox(&mut metadata_enabled, "Look up release metadata on import")
            .changed()
        {
            self.data.use_musicbrainz = metadata_enabled;
            self.persist();
        }
        ui.label(
            RichText::new("MusicBrainz receives the imported track title and artist. OpenMix identifies itself with its public project URL in the required User-Agent.")
                .size(11.0)
                .color(theme_quiet()),
        );
        ui.add_enabled_ui(
            self.data.use_musicbrainz && !self.data.tracks.is_empty(),
            |ui| {
                if secondary_button(ui, "Retry metadata lookup for library").clicked() {
                    let tracks = self.data.tracks.clone();
                    self.lookup_metadata(tracks);
                }
            },
        );
        ui.label(
            RichText::new("Enable lookups above to retry metadata for tracks already imported.")
                .size(11.0)
                .color(theme_quiet()),
        );
        ui.add_space(30.0);
        ui.separator();
        ui.add_space(18.0);
        ui.label(
            RichText::new("Playback")
                .size(17.0)
                .strong()
                .color(theme_text()),
        );
        ui.horizontal(|ui| {
            ui.label("Crossfade length");
            ui.label(RichText::new("6 to 22 seconds").color(theme_soft()));
        });
        ui.label(RichText::new("Drop timing uses whole-track energy analysis and phrase-aligned BPM. No lyrics are fetched or sent. If online lookups or NVIDIA NIM are unavailable, the local audio plan still works.").size(12.0).color(theme_soft()));
        ui.add_space(24.0);
        ui.label(
            RichText::new("Keyboard shortcuts")
                .size(15.0)
                .strong()
                .color(theme_text()),
        );
        for (key, action) in [
            ("Space", "Play or pause"),
            ("Ctrl K", "Search your library"),
            ("Ctrl F", "Filter the library"),
            ("Alt →", "Next track"),
            ("Alt ←", "Previous track"),
            ("Ctrl S", "Shuffle"),
            ("Ctrl R", "Cycle repeat"),
        ] {
            ui.horizontal(|ui| {
                ui.label(RichText::new(key).monospace().color(theme_text()));
                ui.add_space(16.0);
                ui.label(RichText::new(action).color(theme_soft()));
            });
        }
        ui.add_space(26.0);
        ui.separator();
        ui.add_space(12.0);
    }

    fn section_header(&mut self, ui: &mut egui::Ui, title: &str, action: Option<(&str, Page)>) {
        ui.horizontal(|ui| {
            ui.label(RichText::new(title).size(18.0).strong().color(theme_text()));
            if let Some((name, page)) = action {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.link(RichText::new(name).color(theme_soft())).clicked() {
                        self.page = page;
                    }
                });
            }
        });
    }

    fn empty_state(&mut self, ui: &mut egui::Ui, title: &str, detail: &str) {
        let height = ui.available_size_before_wrap().y;
        ui.set_min_height(height);
        ui.add_space(((height - 250.0) * 0.5).max(20.0));
        ui.vertical_centered(|ui| {
            empty_library_mark(ui);
            ui.add_space(12.0);
            ui.label(RichText::new(title).size(21.0).strong().color(theme_text()));
            ui.add_space(5.0);
            ui.label(RichText::new(detail).size(13.0).color(theme_soft()));
            ui.add_space(16.0);
            if self.page == Page::Playlist {
                if primary_button(ui, "Browse library").clicked() {
                    self.page = Page::Library;
                }
            } else if primary_button(ui, "Add local music").clicked() {
                self.import_files();
            }
        });
    }

    fn toggle_like(&mut self, id: &str) {
        if let Some(track) = self.data.tracks.iter_mut().find(|t| t.id == id) {
            track.liked = !track.liked;
        }
        self.persist();
    }

    fn playlist_window(&mut self, ctx: &egui::Context) {
        if self.playlist_dialog {
            let mut open = true;
            egui::Window::new(if self.playlist_edit.is_some() {
                "Rename playlist"
            } else {
                "New playlist"
            })
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label("Give this playlist a name.");
                let response = ui
                    .add(egui::TextEdit::singleline(&mut self.playlist_name).desired_width(300.0));
                response.request_focus();
                ui.horizontal(|ui| {
                    if secondary_button(ui, "Cancel").clicked() {
                        self.playlist_dialog = false;
                    }
                    if ui
                        .add_enabled_ui(!self.playlist_name.trim().is_empty(), |ui| {
                            primary_button(ui, "Save playlist")
                        })
                        .inner
                        .clicked()
                    {
                        let name = self.playlist_name.trim().to_owned();
                        if let Some(id) = self.playlist_edit.take() {
                            if let Some(p) = self.data.playlists.iter_mut().find(|p| p.id == id) {
                                p.name = name;
                                if !self.selected_tracks.is_empty() {
                                    for track_id in &self.selected_tracks {
                                        if !p.tracks.contains(track_id) {
                                            p.tracks.push(track_id.clone());
                                        }
                                    }
                                }
                            }
                        } else {
                            let id =
                                format!("playlist-{}-{}", now_seconds(), self.data.playlists.len());
                            let mut tracks = Vec::new();
                            if !self.selected_tracks.is_empty() {
                                tracks = self.selected_tracks.iter().cloned().collect();
                            }
                            self.data.playlists.push(Playlist {
                                id: id.clone(),
                                name,
                                tracks,
                                created: now_seconds(),
                            });
                            self.active_playlist = Some(id);
                            self.page = Page::Playlist;
                            self.sort = SortMode::PlaylistOrder;
                            self.sort_ascending = true;
                        }
                        self.persist();
                        self.playlist_dialog = false;
                    }
                });
            });
            if !open {
                self.playlist_dialog = false;
            }
        }
        if let Some(id) = self.playlist_delete.clone() {
            egui::Window::new("Delete playlist?")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(
                        "This removes the playlist and keeps your local files in the library.",
                    );
                    ui.horizontal(|ui| {
                        if secondary_button(ui, "Cancel").clicked() {
                            self.playlist_delete = None;
                        }
                        if danger_button(ui, "Delete playlist").clicked() {
                            self.data.playlists.retain(|p| p.id != id);
                            if self.active_playlist.as_deref() == Some(&id) {
                                self.active_playlist = None;
                                self.page = Page::Library;
                            }
                            self.playlist_delete = None;
                            self.persist();
                        }
                    });
                });
        }
    }

    fn notification_toast(&mut self, ctx: &egui::Context) {
        let Some(toast) = &self.toast else {
            return;
        };
        if Instant::now() >= toast.expires_at {
            self.toast = None;
            return;
        }
        let toast = Toast {
            title: toast.title.clone(),
            message: toast.message.clone(),
            error: toast.error,
            expires_at: toast.expires_at,
        };
        let accent = if toast.error {
            theme_red()
        } else {
            Color32::from_rgb(119, 190, 143)
        };
        egui::Area::new(egui::Id::unique("openmix_notification"))
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::RIGHT_TOP, Vec2::new(-18.0, 18.0))
            .show(ctx, |ui| {
                Frame::new()
                    .fill(theme_surface_2())
                    .stroke(Stroke::new(1.0, theme_edge()))
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::symmetric(14, 11))
                    .show(ui, |ui| {
                        ui.set_max_width(380.0);
                        ui.horizontal(|ui| {
                            let (dot, _) =
                                ui.allocate_exact_size(Vec2::splat(8.0), egui::Sense::hover());
                            ui.painter().circle_filled(dot.center(), 4.0, accent);
                            ui.vertical(|ui| {
                                ui.label(
                                    RichText::new(&toast.title)
                                        .size(13.0)
                                        .strong()
                                        .color(theme_text()),
                                );
                                ui.label(
                                    RichText::new(&toast.message).size(11.0).color(theme_soft()),
                                );
                            });
                        });
                    });
            });
    }
}

impl eframe::App for OpenMix {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.receive_messages();
        self.update_shortcuts(&ctx);
        if self.current.is_some() && self.audio.ended() {
            if self.repeat == 1 {
                if let Some(id) = self.current.clone() {
                    self.play_id(&id);
                }
            } else if !self.queue.is_empty() || self.repeat == 2 {
                self.next();
            }
        }
        self.player_bar(ui);
        self.left_sidebar(ui);
        self.right_now_panel(ui);
        self.content(ui);
        self.playlist_window(&ctx);
        self.notification_toast(&ctx);
        ctx.request_repaint_after(Duration::from_millis(33));
    }
}

fn record_art(ui: &mut egui::Ui, track: &Track, size: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, CornerRadius::same(4), Color32::from_rgb(36, 38, 41));
    let texture = track.cover_art_path.as_ref().and_then(|path| {
        COVER_TEXTURES.with(|textures| {
            let mut textures = textures.borrow_mut();
            if let Some(texture) = textures.get(path) {
                return Some(texture.clone());
            }
            let rgba = image::open(path).ok()?.to_rgba8();
            let dimensions = [rgba.width() as usize, rgba.height() as usize];
            let color_image = egui::ColorImage::from_rgba_unmultiplied(dimensions, rgba.as_raw());
            let texture = ui.ctx().load_texture(
                format!("cover:{}", path.display()),
                color_image,
                egui::TextureOptions::LINEAR,
            );
            textures.insert(path.clone(), texture.clone());
            Some(texture)
        })
    });
    if let Some(texture) = texture {
        painter.image(
            texture.id(),
            rect,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        return;
    }
    let center = rect.center();
    let r = size * 0.36;
    painter.circle_stroke(center, r, Stroke::new(1.0, Color32::from_rgb(83, 85, 87)));
    painter.circle_stroke(
        center,
        r * 0.61,
        Stroke::new(1.0, Color32::from_rgb(65, 67, 69)),
    );
    painter.circle_filled(center, size * 0.065, theme_red());
}

fn mix_score(from: &Track, to: &Track) -> f32 {
    let bpm_gap = if from.bpm == 0 || to.bpm == 0 {
        10.0
    } else {
        let a = (from.bpm as f32 - to.bpm as f32).abs();
        a.min((from.bpm as f32 - 2.0 * to.bpm as f32).abs())
            .min((2.0 * from.bpm as f32 - to.bpm as f32).abs())
    };
    bpm_gap * 1.7 + (from.energy - to.energy).abs() * 20.0
}

fn bpm_label(bpm: u16) -> String {
    if bpm == 0 {
        "—".into()
    } else {
        bpm.to_string()
    }
}
fn format_time(seconds: u32) -> String {
    format!("{}:{:02}", seconds / 60, seconds % 60)
}
fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn shuffle_slice(ids: &mut [String]) {
    let mut seed = now_seconds() as usize ^ ids.len().wrapping_mul(7919);
    for i in (1..ids.len()).rev() {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ids.swap(i, seed % (i + 1));
    }
}

#[derive(Serialize, Deserialize)]
struct MixInfo {
    status: String,
    total_tracks: usize,
    completed_tracks: usize,
    track_ids: Vec<String>,
    output_path: PathBuf,
    updated_at: u64,
    #[serde(default)]
    error: Option<String>,
}

/// The directory the running binary lives in, falling back to the current
/// directory if it can't be determined (e.g. the executable was removed
/// out from under the process).
fn binary_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Progress for the in-flight "mix all" render, written next to the binary
/// so the render's state lives on disk rather than only in memory: a crash
/// or restart mid-render leaves both this file and the partial WAV behind
/// instead of losing everything.
fn mixinfo_path() -> PathBuf {
    binary_dir().join(".mixinfo")
}

fn write_mixinfo(info: &MixInfo) {
    let Ok(json) = serde_json::to_vec_pretty(info) else {
        return;
    };
    let _ = fs::write(mixinfo_path(), json);
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1480.0, 940.0])
            .with_min_inner_size([980.0, 680.0])
            .with_icon(app_icon()),
        ..Default::default()
    };
    eframe::run_native(
        "OpenMix",
        options,
        Box::new(|cc| Ok(Box::new(OpenMix::new(cc)))),
    )
}

fn app_icon() -> Arc<egui::IconData> {
    let size = 64u32;
    let radius = 14.0f32;
    let bars = [(17, 17), (25, 29), (33, 39), (41, 27), (49, 17)];
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            let dx = if px < radius {
                radius - px
            } else if px > 64.0 - radius {
                px - (64.0 - radius)
            } else {
                0.0
            };
            let dy = if py < radius {
                radius - py
            } else if py > 64.0 - radius {
                py - (64.0 - radius)
            } else {
                0.0
            };
            let inside = dx * dx + dy * dy <= radius * radius;
            let wave = bars.iter().any(|(bar_x, height)| {
                x >= *bar_x && x < *bar_x + 4 && (py - 32.0).abs() <= *height as f32 * 0.5
            });
            let color = if wave { theme_red() } else { theme_bg() };
            rgba.extend_from_slice(&[
                color.r(),
                color.g(),
                color.b(),
                if inside { 255 } else { 0 },
            ]);
        }
    }
    Arc::new(egui::IconData {
        rgba,
        width: size,
        height: size,
    })
}
