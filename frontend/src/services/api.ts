import {
  HealthResponse,
  MetadataResponse,
  BenchmarkResponse,
  SignaturesResponse,
  AnalysisResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function fetchMetadata(): Promise<MetadataResponse> {
  const res = await fetch(`${API_BASE}/api/v1/metadata`);
  if (!res.ok) throw new Error(`Failed to fetch metadata: ${res.statusText}`);
  return res.json();
}

export async function fetchBenchmark(): Promise<BenchmarkResponse> {
  const res = await fetch(`${API_BASE}/api/v1/research/benchmark`);
  if (!res.ok) throw new Error(`Failed to fetch benchmark: ${res.statusText}`);
  return res.json();
}

export async function fetchSignatures(): Promise<SignaturesResponse> {
  const res = await fetch(`${API_BASE}/api/v1/research/signatures`);
  if (!res.ok) throw new Error(`Failed to fetch signatures: ${res.statusText}`);
  return res.json();
}

export async function analyzeSequence(unitId: number, sequence: any[]): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/api/v1/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ unit_id: unitId, sequence }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Analysis failed with status ${res.status}`);
  }
  return res.json();
}

export async function analyzeCSV(file: File): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/api/v1/analyze/csv`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `CSV analysis failed with status ${res.status}`);
  }
  return res.json();
}
