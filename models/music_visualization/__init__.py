"""Music visualization: waveforms, spectrograms, harmonic circles, piano rolls, rhythm, analysis."""
from .waveform import WaveformRegion, WaveformRenderer, WaveformThumbnail
from .spectrogram import (
    SpectrogramRenderer,
    MelSpectrogram,
    LogSpectrogram,
    SpectrogramColormap,
)
from .harmonic_circle import (
    CircleOfFifths,
    ChromaWheel,
    TonnetzProjection,
    KeyRegion,
)
from .piano_roll import (
    NoteLabel,
    VelocityColorMap,
    PianoRollRenderer,
    PianoRollOverlay,
)
from .rhythm_viz import RhythmGrid, OnsetPlot, TempoGraph, GrooveMap
from .analysis_plots import (
    LoudnessCurve,
    DynamicRangePlot,
    FrequencyResponse,
    StereoFieldPlot,
)

__all__ = [
    "WaveformRegion", "WaveformRenderer", "WaveformThumbnail",
    "SpectrogramRenderer", "MelSpectrogram", "LogSpectrogram", "SpectrogramColormap",
    "CircleOfFifths", "ChromaWheel", "TonnetzProjection", "KeyRegion",
    "NoteLabel", "VelocityColorMap", "PianoRollRenderer", "PianoRollOverlay",
    "RhythmGrid", "OnsetPlot", "TempoGraph", "GrooveMap",
    "LoudnessCurve", "DynamicRangePlot", "FrequencyResponse", "StereoFieldPlot",
    "WaveformData", "compute_waveform_display",
    "compute_spectrogram_display", "compute_chromagram_display",
    "loudness_curve", "stereo_correlation_curve", "phase_correlation_val",
    "frequency_spectrum_snapshot", "color_map_note",
    "GraphNode", "GraphEdge", "NetworkGraph", "force_directed_layout",
    "chord_transition_graph", "genre_taxonomy_graph", "artist_similarity_graph",
    "community_detection_louvain",
    "TonnetzPoint", "TonnetzEdge", "ChordRegion",
    "tonnetz_viz_coordinates", "chord_viz_centroid",
    "chord_trajectory", "tonnetz_viz_distance",
    "tonnetz_grid", "neo_riemannian_edges",
    "harmonic_function_color", "region_data",
    "CirclePoint", "ArcPath", "FIFTHS_ORDER",
    "circle_of_fifths_layout", "chromatic_circle_layout",
    "pitch_class_distribution", "key_region_highlight",
    "chord_progression_arcs", "interval_vector_radial",
    "fifths_distance", "angular_distance",
    "FormSection", "FormTree", "COMMON_FORMS",
    "section_boundaries", "build_form_tree",
    "repetition_matrix", "novelty_curve",
    "section_similarity_matrix", "classify_form",
    "AnnotationType", "Annotation", "HarmonicSeries", "PitchContour", "FormantTrack",
    "harmonic_series_annotations", "onset_annotations", "section_boundary_annotations",
    "pitch_contour_annotations", "formant_annotations", "note_region_annotations",
    "spectral_peak_annotations", "noise_band_annotations",
    "merge_annotations", "filter_annotations", "annotation_density",
    "EasingType", "Keyframe", "AnimationTrack", "Timeline", "BeatSync",
    "ease", "ease_array", "crossfade",
    "lfo_track", "beat_reactive_track", "onset_reactive_track",
    "spectrum_band_tracks",
    "SRConfig", "stft_magnitude", "amplitude_to_db",
    "hz_to_mel", "mel_to_hz", "mel_filterbank", "sr_mel_spectrogram",
    "cqt_kernel", "constant_q_transform", "sr_chromagram",
    "sr_spectral_flux", "onset_strength", "spectral_contrast",
    "interpolate_keyframes", "keyframes_from_features",
    "snap_to_beats", "beat_sync_curve",
    "WaveformGeometry", "waveform_display_data", "band_energy_series",
    "SpectrogramTile", "spectrogram_tiles",
    "ColorMapping", "feature_to_color",
    "CameraKeyframe", "camera_path",
]

from .waveform_display import (
    WaveformData, compute_waveform_display,
    compute_spectrogram_display, compute_chromagram_display,
    loudness_curve,
    stereo_correlation as stereo_correlation_curve,
    phase_correlation as phase_correlation_val,
    frequency_spectrum_snapshot, color_map_note,
)
from .chromagram_display import (
    compute_chromagram, hpcp, chroma_to_tonnetz,
    chroma_correlation, key_profile_correlation,
    Chromagram, TonnetzCoords, PITCH_CLASSES,
)
from .network_graph import (
    GraphNode, GraphEdge, NetworkGraph, force_directed_layout,
    chord_transition_graph, genre_taxonomy_graph, artist_similarity_graph,
    community_detection_louvain,
)
from .tonnetz_viz import (
    TonnetzPoint, TonnetzEdge, ChordRegion,
    tonnetz_coordinates as tonnetz_viz_coordinates,
    chord_tonnetz_centroid as chord_viz_centroid,
    chord_trajectory, tonnetz_distance as tonnetz_viz_distance,
    tonnetz_grid, neo_riemannian_edges,
    harmonic_function_color, region_data,
)
from .circular_viz import (
    CirclePoint, ArcPath, FIFTHS_ORDER,
    circle_of_fifths_layout, chromatic_circle_layout,
    pitch_class_distribution, key_region_highlight,
    chord_progression_arcs, interval_vector_radial,
    fifths_distance, angular_distance,
)
from .form_diagram import (
    FormSection, FormTree, COMMON_FORMS,
    section_boundaries, build_form_tree,
    repetition_matrix, novelty_curve,
    section_similarity_matrix, classify_form,
)
from .spectrogram_annotation import (
    AnnotationType, Annotation, HarmonicSeries, PitchContour, FormantTrack,
    harmonic_series_annotations, onset_annotations,
    section_boundary_annotations, pitch_contour_annotations,
    formant_annotations, note_region_annotations,
    spectral_peak_annotations, noise_band_annotations,
    merge_annotations, filter_annotations, annotation_density,
)
from .animation_timeline import (
    EasingType, Keyframe, AnimationTrack, Timeline, BeatSync,
    ease, ease_array, crossfade,
    lfo_track, beat_reactive_track, onset_reactive_track,
    spectrum_band_tracks,
)
from .animation_timeline import (
    EasingType, ease, ease_array, Keyframe, AnimationTrack, BeatSync,
    Timeline, beat_reactive_track, onset_reactive_track,
    spectrum_band_tracks, lfo_track, crossfade,
)
from .spectrogram_annotation import (
    AnnotationType, Annotation, HarmonicSeries, PitchContour, FormantTrack,
    harmonic_series_annotations, onset_annotations, section_boundary_annotations,
    pitch_contour_annotations, formant_annotations, note_region_annotations,
    spectral_peak_annotations, noise_band_annotations,
    merge_annotations, filter_annotations, annotation_density,
)
from .spectrogram_renderer import (
    SpectrogramConfig as SRConfig,
    stft_magnitude, amplitude_to_db, hz_to_mel, mel_to_hz,
    mel_filterbank, mel_spectrogram as sr_mel_spectrogram,
    cqt_kernel, constant_q_transform,
    chromagram as sr_chromagram,
    spectral_flux as sr_spectral_flux,
    onset_strength, spectral_contrast,
)
from .animation_data import (
    interpolate_keyframes, keyframes_from_features,
    snap_to_beats, beat_sync_curve,
    WaveformGeometry, waveform_display_data, band_energy_series,
    SpectrogramTile, spectrogram_tiles,
    ColorMapping, feature_to_color,
    CameraKeyframe, camera_path,
)
