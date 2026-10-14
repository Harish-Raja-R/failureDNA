export interface HealthResponse {
  status: 'healthy' | 'degraded';
  version: string;
  is_ready: boolean;
  initialized_at: string | null;
  error: string | null;
  artifacts: {
    frozen_checkpoint_exists: boolean;
    encoder_model_loaded: boolean;
    preprocessor_loaded: boolean;
    ridge_probe_loaded: boolean;
    signature_clusterer_loaded: boolean;
  };
}

export interface MetadataResponse {
  project_name: string;
  version: string;
  dataset: string;
  operating_regimes_count: number;
  sensor_channels_count: number;
  setting_channels_count: number;
  required_features: string[];
  sequence_window_length: number;
  padding_mode: string;
  model_architecture: {
    encoder: string;
    checkpoint_sha256: string;
    latent_dim: number;
    downstream_estimator: string;
    signature_clusterer: string;
  };
  disclaimer: string;
}

export interface SensorTrend {
  start: number;
  end: number;
  min: number;
  max: number;
  mean: number;
  delta: number;
}

export interface AnalysisResponse {
  unit_id: number;
  sequence_length: number;
  window_size: number;
  padding_applied: boolean;
  padded_cycles: number;
  predicted_rul: number;
  predicted_rul_unclipped: number;
  rul_cap_applied: boolean;
  signature_state: number;
  signature_name: string;
  signature_description: string;
  distance_to_centroid: number | null;
  detected_operating_regimes: number[];
  sensor_trends: Record<string, SensorTrend>;
  warnings: string[];
  timestamp: string;
  model_metadata: {
    encoder: string;
    estimator: string;
    checkpoint_sha256: string;
  };
}

export interface ModelComparisonRow {
  model_key: string;
  model_name: string;
  category: string;
  features: string;
  mae: number;
  mae_ci_lower: number;
  mae_ci_upper: number;
  rmse: number;
  rmse_ci_lower: number;
  rmse_ci_upper: number;
  r2: number;
  r2_ci_lower: number;
  r2_ci_upper: number;
  nasa_score: number;
  nasa_ci_lower: number;
  nasa_ci_upper: number;
  fit_time_sec: number;
}

export interface PairedBootstrapRow {
  comparison: string;
  metric: string;
  delta_mean: number;
  ci_lower: number;
  ci_upper: number;
  raw_p_value: number;
  p_davison_hinkley: number;
  holm_adj_p_metric: number;
  holm_adj_p_primary: number;
  holm_adj_p_omnibus: number;
  sig_omnibus_05: boolean;
  stat_advantage_winner: string;
}

export interface BenchmarkResponse {
  benchmark_dataset: string;
  test_engines_count: number;
  observations_count: number;
  summary: any;
  model_comparison: ModelComparisonRow[];
  statistical_audit_paired_bootstrap: PairedBootstrapRow[];
  scientific_conclusion: string;
}

export interface SignaturesResponse {
  benchmark_cohort: string;
  clusters_count: number;
  cluster_order_mapping: Record<string, number>;
  fleet_occupancy: Record<string, { count: number; percentage: number }>;
  final_window_distribution: Record<string, number>;
  transition_matrix: number[][];
  empirical_mean_dwell_times: Record<string, number>;
  signature_regime_nmi: number;
  hazard_rate_audit: {
    description: string;
    risk_set_cycles: number;
    operational_exits: number;
    reciprocal_hazard: number;
    observed_operational_exit_mean_dwell: number;
    observed_terminal_failure_mean_dwell: number;
    pooled_absorbing_mean_dwell: number;
  };
  methodological_note: string;
}
