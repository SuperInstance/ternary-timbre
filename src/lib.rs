//! # ternary-timbre
//!
//! Timbre analysis for agent output characterization. In music, timbre is what
//! makes a trumpet sound different from a violin even when playing the same note.
//! This crate applies the same concept to AI agents: what makes one agent's output
//! recognizably different from another's, even when they're solving the same task?

use std::collections::HashMap;

/// A spectral fingerprint of an agent's output style.
///
/// Just as musical timbre is described by harmonics, attack, decay, and spectral
/// envelope, a TimbreProfile captures the statistical features that characterize
/// how an agent communicates.
#[derive(Debug, Clone)]
pub struct TimbreProfile {
    /// The agent this profile describes.
    pub agent_id: String,
    /// Ratio of output variation to average — how "rhythmic" the output pattern is.
    /// High rhythm ratio = output lengths vary a lot (bursty).
    /// Low rhythm ratio = consistent output lengths (steady).
    pub rhythm_ratio: f64,
    /// Range of output "loudness" (content density / informativeness).
    /// High dynamic range = agent varies between terse and verbose.
    /// Low dynamic range = agent is consistently verbose or terse.
    pub dynamic_range: f64,
    /// How often the agent deviates from its expected pattern.
    /// High syncopation = unpredictable, creative outputs.
    /// Low syncopation = predictable, formulaic outputs.
    pub syncopation_index: f64,
    /// Average information density (unique concepts per unit length).
    pub density: f64,
    /// Sample outputs used to build this profile.
    pub sample_count: usize,
}

impl TimbreProfile {
    /// Build a profile from a collection of output strings.
    pub fn from_outputs(agent_id: impl Into<String>, outputs: &[&str]) -> Self {
        if outputs.is_empty() {
            return Self {
                agent_id: agent_id.into(),
                rhythm_ratio: 0.0,
                dynamic_range: 0.0,
                syncopation_index: 0.0,
                density: 0.0,
                sample_count: 0,
            };
        }

        let lengths: Vec<usize> = outputs.iter().map(|s| s.len()).collect();
        let densities: Vec<f64> = outputs.iter().map(|s| compute_density(s)).collect();

        let rhythm_ratio = compute_rhythm_ratio(&lengths);
        let dynamic_range = compute_dynamic_range(&densities);
        let syncopation_index = compute_syncopation(&lengths);
        let density = densities.iter().sum::<f64>() / densities.len() as f64;

        Self {
            agent_id: agent_id.into(),
            rhythm_ratio,
            dynamic_range,
            syncopation_index,
            density,
            sample_count: outputs.len(),
        }
    }

    /// Create a profile with manually specified features.
    pub fn from_features(
        agent_id: impl Into<String>,
        rhythm_ratio: f64,
        dynamic_range: f64,
        syncopation_index: f64,
        density: f64,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            rhythm_ratio,
            dynamic_range,
            syncopation_index,
            density,
            sample_count: 0,
        }
    }

    /// Convert the profile to a feature vector for distance calculations.
    pub fn feature_vector(&self) -> Vec<f64> {
        vec![
            self.rhythm_ratio,
            self.dynamic_range,
            self.syncopation_index,
            self.density,
        ]
    }
}

/// Compute information density: ratio of unique words to total length.
fn compute_density(text: &str) -> f64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }
    let unique: std::collections::HashSet<&str> = words.iter().copied().collect();
    unique.len() as f64 / words.len() as f64
}

/// Compute rhythm ratio: coefficient of variation of output lengths.
fn compute_rhythm_ratio(lengths: &[usize]) -> f64 {
    if lengths.len() < 2 {
        return 0.0;
    }
    let mean = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
    if mean == 0.0 {
        return 0.0;
    }
    let variance = lengths
        .iter()
        .map(|&l| (l as f64 - mean).powi(2))
        .sum::<f64>()
        / lengths.len() as f64;
    variance.sqrt() / mean
}

/// Compute dynamic range: max density - min density.
fn compute_dynamic_range(densities: &[f64]) -> f64 {
    if densities.is_empty() {
        return 0.0;
    }
    let max = densities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = densities.iter().cloned().fold(f64::INFINITY, f64::min);
    max - min
}

/// Compute syncopation index: how much consecutive outputs deviate from the expected trend.
fn compute_syncopation(lengths: &[usize]) -> f64 {
    if lengths.len() < 3 {
        return 0.0;
    }
    let mut deviations = 0usize;
    for i in 2..lengths.len() {
        // Expected: linear trend from i-2 to i-1 extended to i
        let trend = (lengths[i - 1] as f64 + (lengths[i - 1] as f64 - lengths[i - 2] as f64)) as isize;
        let actual = lengths[i] as isize;
        if (actual - trend).unsigned_abs() > (trend.unsigned_abs() as f64 * 0.3) as usize {
            deviations += 1;
        }
    }
    deviations as f64 / (lengths.len() - 2) as f64
}

/// Measures the stylistic distance between two agents' output profiles.
///
/// Uses Euclidean distance in the normalized feature space.
#[derive(Debug, Clone)]
pub struct TimbreDistance {
    /// Euclidean distance between feature vectors.
    pub distance: f64,
    /// Breakdown by feature dimension.
    pub breakdown: HashMap<String, f64>,
}

impl TimbreDistance {
    /// Compute the distance between two profiles.
    pub fn between(a: &TimbreProfile, b: &TimbreProfile) -> Self {
        let va = a.feature_vector();
        let vb = b.feature_vector();

        let labels = ["rhythm", "dynamic_range", "syncopation", "density"];
        let mut breakdown = HashMap::new();
        let mut sum_sq = 0.0;

        for (i, &label) in labels.iter().enumerate() {
            let diff = va[i] - vb[i];
            breakdown.insert(label.to_string(), diff.abs());
            sum_sq += diff * diff;
        }

        Self {
            distance: sum_sq.sqrt(),
            breakdown,
        }
    }

    /// Whether two profiles are "similar" (within a threshold).
    pub fn is_similar(&self, threshold: f64) -> bool {
        self.distance < threshold
    }
}

/// Groups agents by output style similarity using hierarchical clustering.
pub struct StyleCluster {
    /// Cluster label → list of agent IDs.
    clusters: HashMap<String, Vec<String>>,
    /// Distance threshold for grouping.
    threshold: f64,
}

impl StyleCluster {
    /// Create a new clusterer with a distance threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            clusters: HashMap::new(),
            threshold,
        }
    }

    /// Cluster agents based on their profiles.
    pub fn fit(&mut self, profiles: &[TimbreProfile]) {
        self.clusters.clear();

        if profiles.is_empty() {
            return;
        }

        // Simple single-linkage agglomerative clustering
        let n = profiles.len();
        let mut assigned: Vec<Option<usize>> = vec![None; n];

        // Build distance matrix
        let mut distances = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = TimbreDistance::between(&profiles[i], &profiles[j]);
                distances[i][j] = d.distance;
                distances[j][i] = d.distance;
            }
        }

        // Assign clusters
        let mut cluster_id = 0;
        for i in 0..n {
            if assigned[i].is_some() {
                continue;
            }

            // Find all agents within threshold
            let mut members = vec![i];
            assigned[i] = Some(cluster_id);

            for j in (i + 1)..n {
                if assigned[j].is_none() {
                    // Check if j is within threshold of any current member
                    let close = members.iter().any(|&m| distances[m][j] < self.threshold);
                    if close {
                        members.push(j);
                        assigned[j] = Some(cluster_id);
                    }
                }
            }

            cluster_id += 1;
        }

        // Build cluster map
        for (i, profile) in profiles.iter().enumerate() {
            let cid = assigned[i].unwrap();
            let label = format!("cluster_{}", cid);
            self.clusters
                .entry(label)
                .or_insert_with(Vec::new)
                .push(profile.agent_id.clone());
        }
    }

    /// Get all cluster labels.
    pub fn cluster_labels(&self) -> Vec<&str> {
        self.clusters.keys().map(|s| s.as_str()).collect()
    }

    /// Get members of a specific cluster.
    pub fn members(&self, label: &str) -> Vec<&str> {
        self.clusters
            .get(label)
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Number of clusters.
    pub fn len(&self) -> usize {
        self.clusters.len()
    }

    /// Whether there are no clusters.
    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }

    /// Find which cluster an agent belongs to.
    pub fn find_cluster(&self, agent_id: &str) -> Option<&str> {
        for (label, members) in &self.clusters {
            if members.iter().any(|m| m == agent_id) {
                return Some(label.as_str());
            }
        }
        None
    }
}

/// Tracks how an agent's output style changes over time.
///
/// Like watching a musician's style evolve across albums, this captures
/// trends, shifts, and developments in an agent's communication patterns.
pub struct TimbreEvolution {
    /// Ordered snapshots of the agent's profile over time.
    snapshots: Vec<TimbreProfile>,
    /// The agent being tracked.
    agent_id: String,
}

impl TimbreEvolution {
    /// Create a new evolution tracker for an agent.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            snapshots: Vec::new(),
            agent_id: agent_id.into(),
        }
    }

    /// Add a new profile snapshot.
    pub fn add_snapshot(&mut self, profile: TimbreProfile) {
        self.snapshots.push(profile);
    }

    /// Get all snapshots in chronological order.
    pub fn snapshots(&self) -> &[TimbreProfile] {
        &self.snapshots
    }

    /// Measure the total drift from first to latest snapshot.
    pub fn total_drift(&self) -> f64 {
        if self.snapshots.len() < 2 {
            return 0.0;
        }
        TimbreDistance::between(&self.snapshots[0], self.snapshots.last().unwrap()).distance
    }

    /// Measure the drift between consecutive snapshots.
    pub fn step_drifts(&self) -> Vec<f64> {
        self.snapshots
            .windows(2)
            .map(|w| TimbreDistance::between(&w[0], &w[1]).distance)
            .collect()
    }

    /// Detect sudden style changes (steps with drift above a threshold).
    pub fn detect_shifts(&self, threshold: f64) -> Vec<usize> {
        self.step_drifts()
            .iter()
            .enumerate()
            .filter(|(_, d)| **d > threshold)
            .map(|(i, _)| i + 1) // Index of the snapshot after the shift
            .collect()
    }

    /// Get the dominant trend for each feature.
    pub fn trends(&self) -> HashMap<String, TrendDirection> {
        if self.snapshots.len() < 2 {
            return HashMap::new();
        }

        let first = &self.snapshots[0];
        let last = self.snapshots.last().unwrap();
        let labels = ["rhythm_ratio", "dynamic_range", "syncopation_index", "density"];
        let first_v = first.feature_vector();
        let last_v = last.feature_vector();

        labels
            .iter()
            .zip(first_v.iter().zip(&last_v))
            .map(|(&label, (&f, &l))| {
                let diff = l - f;
                let direction = if diff > 0.05 {
                    TrendDirection::Increasing
                } else if diff < -0.05 {
                    TrendDirection::Decreasing
                } else {
                    TrendDirection::Stable
                };
                (label.to_string(), direction)
            })
            .collect()
    }
}

/// Direction of a trend in a feature over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Feature is increasing over time.
    Increasing,
    /// Feature is decreasing over time.
    Decreasing,
    /// Feature is relatively stable.
    Stable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_style_low_distance() {
        let outputs_a = vec!["Hello world this is a test", "Hello world another test"];
        let outputs_b = vec!["Hello world this is also a test", "Hello world yet another test"];

        let profile_a = TimbreProfile::from_outputs("agent-a", &outputs_a);
        let profile_b = TimbreProfile::from_outputs("agent-b", &outputs_b);

        let distance = TimbreDistance::between(&profile_a, &profile_b);
        assert!(
            distance.distance < 0.3,
            "Similar styles should have low distance, got {}",
            distance.distance
        );
    }

    #[test]
    fn test_different_style_high_distance() {
        // Verbose, varied agent
        let outputs_a = vec![
            "This is a very long and detailed output with many words and explanations",
            "short",
            "Another extremely verbose output that contains a lot of detailed information",
        ];

        // Terse, consistent agent
        let outputs_b = vec!["ok", "done", "yes"];

        let profile_a = TimbreProfile::from_outputs("verbose-agent", &outputs_a);
        let profile_b = TimbreProfile::from_outputs("terse-agent", &outputs_b);

        let distance = TimbreDistance::between(&profile_a, &profile_b);
        assert!(
            distance.distance > 0.3,
            "Very different styles should have high distance, got {}",
            distance.distance
        );
    }

    #[test]
    fn test_cluster_groups_similar_agents() {
        // Three agents with similar style
        let p1 = TimbreProfile::from_features("agent-a", 0.5, 0.3, 0.2, 0.7);
        let p2 = TimbreProfile::from_features("agent-b", 0.52, 0.31, 0.21, 0.69);
        let p3 = TimbreProfile::from_features("agent-c", 0.48, 0.29, 0.19, 0.71);

        // One agent with very different style
        let p4 = TimbreProfile::from_features("agent-d", 2.0, 1.5, 1.8, 0.1);

        let mut clusterer = StyleCluster::new(0.5);
        clusterer.fit(&[p1, p2, p3, p4]);

        assert_eq!(clusterer.len(), 2, "Should produce 2 clusters");

        // agent-a, agent-b, agent-c should be in the same cluster
        let cluster_a = clusterer.find_cluster("agent-a");
        let cluster_b = clusterer.find_cluster("agent-b");
        let cluster_c = clusterer.find_cluster("agent-c");
        let cluster_d = clusterer.find_cluster("agent-d");

        assert_eq!(cluster_a, cluster_b, "a and b should be in same cluster");
        assert_eq!(cluster_a, cluster_c, "a and c should be in same cluster");
        assert_ne!(cluster_a, cluster_d, "a and d should be in different clusters");
    }

    #[test]
    fn test_evolution_tracks_change() {
        let mut evolution = TimbreEvolution::new("agent-x");

        evolution.add_snapshot(TimbreProfile::from_features("agent-x", 0.5, 0.3, 0.2, 0.7));
        evolution.add_snapshot(TimbreProfile::from_features("agent-x", 0.6, 0.4, 0.3, 0.65));
        evolution.add_snapshot(TimbreProfile::from_features("agent-x", 0.9, 0.7, 0.5, 0.5));

        assert_eq!(evolution.snapshots().len(), 3);
        assert!(evolution.total_drift() > 0.0, "Should have measurable drift");

        let drifts = evolution.step_drifts();
        assert_eq!(drifts.len(), 2);

        let trends = evolution.trends();
        assert_eq!(trends.get("rhythm_ratio"), Some(&TrendDirection::Increasing));
        assert_eq!(trends.get("density"), Some(&TrendDirection::Decreasing));
    }

    #[test]
    fn test_evolution_detect_shifts() {
        let mut evolution = TimbreEvolution::new("agent-y");

        // Gradual change
        evolution.add_snapshot(TimbreProfile::from_features("agent-y", 0.5, 0.5, 0.5, 0.5));
        evolution.add_snapshot(TimbreProfile::from_features("agent-y", 0.52, 0.51, 0.49, 0.5));

        // Sudden shift
        evolution.add_snapshot(TimbreProfile::from_features("agent-y", 2.0, 1.5, 1.8, 0.1));

        let shifts = evolution.detect_shifts(0.5);
        assert!(shifts.contains(&2), "Should detect shift at snapshot 2");
    }

    #[test]
    fn test_feature_extraction_values() {
        let outputs = vec!["hello world test", "hello world another test case"];
        let profile = TimbreProfile::from_outputs("agent", &outputs);

        assert!(profile.rhythm_ratio >= 0.0, "Rhythm ratio should be non-negative");
        assert!(profile.dynamic_range >= 0.0, "Dynamic range should be non-negative");
        assert!(profile.syncopation_index >= 0.0, "Syncopation should be non-negative");
        assert!(profile.density > 0.0 && profile.density <= 1.0, "Density should be in (0, 1]");
        assert_eq!(profile.sample_count, 2);
    }

    #[test]
    fn test_empty_profile() {
        let profile = TimbreProfile::from_outputs("empty", &[]);
        assert_eq!(profile.rhythm_ratio, 0.0);
        assert_eq!(profile.density, 0.0);
        assert_eq!(profile.sample_count, 0);
    }

    #[test]
    fn test_density_computation() {
        // All unique words → density = 1.0
        assert!((compute_density("a b c d") - 1.0).abs() < 0.01);

        // Repeated words → density < 1.0
        assert!(compute_density("hello hello hello") < 0.5);

        // Empty → 0.0
        assert_eq!(compute_density(""), 0.0);
    }

    #[test]
    fn test_rhythm_ratio() {
        // Constant lengths → low rhythm ratio
        let consistent = vec![100, 100, 100, 100];
        assert!(compute_rhythm_ratio(&consistent) < 0.01);

        // Variable lengths → high rhythm ratio
        let varied = vec![10, 200, 5, 300];
        assert!(compute_rhythm_ratio(&varied) > 0.5);
    }

    #[test]
    fn test_distance_breakdown() {
        let a = TimbreProfile::from_features("a", 0.5, 0.5, 0.5, 0.5);
        let b = TimbreProfile::from_features("b", 0.5, 0.5, 0.5, 0.5);

        let dist = TimbreDistance::between(&a, &b);
        assert_eq!(dist.distance, 0.0);
        assert!(dist.is_similar(0.1));
    }

    #[test]
    fn test_cluster_empty() {
        let mut clusterer = StyleCluster::new(0.5);
        clusterer.fit(&[]);
        assert!(clusterer.is_empty());
    }

    #[test]
    fn test_evolution_single_snapshot() {
        let mut evo = TimbreEvolution::new("agent");
        evo.add_snapshot(TimbreProfile::from_features("agent", 0.5, 0.5, 0.5, 0.5));
        assert_eq!(evo.total_drift(), 0.0);
        assert!(evo.step_drifts().is_empty());
    }

    #[test]
    fn test_find_cluster_none() {
        let clusterer = StyleCluster::new(0.5);
        assert_eq!(clusterer.find_cluster("nonexistent"), None);
    }
}
