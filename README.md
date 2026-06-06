# ternary-timbre

**Timbre analysis for agent output characterization.**

In music, timbre is what makes a trumpet sound different from a violin even when they play the exact same note. It's the spectral fingerprint — the overtones, attack, decay, and texture that make each instrument unique. This crate applies that concept to AI agents: what makes one agent's output recognizably different from another's, even when they're solving the same task?

## Why Timbre?

When you're working with multiple AI agents, you need to understand not just *what* they produce, but *how* they produce it. Timbre analysis lets you:

- **Fingerprint agent styles** — capture the statistical signature of how an agent communicates
- **Measure stylistic distance** — quantify how different two agents' output styles are
- **Cluster similar agents** — group agents that communicate in similar ways
- **Track style evolution** — monitor how an agent's output changes over time

## Feature Extraction

Every agent's output style is characterized by four features, inspired by musical timbre descriptors:

### Rhythm Ratio (Coefficient of Variation)
How bursty vs. steady is the agent's output? A high rhythm ratio means the agent alternates between long and short outputs. A low ratio means consistent output lengths.

- **High rhythm ratio**: Agent writes novels, then one-liners, then novels again
- **Low rhythm ratio**: Agent consistently produces similar-length outputs

### Dynamic Range
How much does the agent vary in information density? High dynamic range means the agent swings between terse, dense outputs and verbose, redundant ones.

- **High dynamic range**: Agent is sometimes extremely concise, sometimes very wordy
- **Low dynamic range**: Agent maintains consistent information density

### Syncopation Index
How unpredictable is the agent's output pattern? High syncopation means the agent deviates from its established trends — it's creative, surprising, or inconsistent.

- **High syncopation**: Agent's output lengths don't follow predictable patterns
- **Low syncopation**: Agent's output follows smooth, predictable trends

### Density
Average information density — the ratio of unique words to total words. High density means every word carries information. Low density means repetition and filler.

## Core Types

### `TimbreProfile`

The spectral fingerprint of an agent's output style.

```rust
use ternary_timbre::TimbreProfile;

let outputs = vec![
    "The quantum algorithm runs in O(n log n) time complexity",
    "Results confirm theoretical predictions with 97.3% accuracy",
    "Further analysis reveals a linear correlation (r=0.94)",
];

let profile = TimbreProfile::from_outputs("gpt-4-researcher", &outputs);
println!("Rhythm: {:.3}", profile.rhythm_ratio);
println!("Dynamic range: {:.3}", profile.dynamic_range);
println!("Syncopation: {:.3}", profile.syncopation_index);
println!("Density: {:.3}", profile.density);
```

### `TimbreDistance`

Measures the stylistic distance between two agents' profiles using Euclidean distance in the normalized feature space.

```rust
use ternary_timbre::{TimbreProfile, TimbreDistance};

let profile_a = TimbreProfile::from_outputs("agent-a", &["hello world test"]);
let profile_b = TimbreProfile::from_outputs("agent-b", &["hello world another test"]);

let distance = TimbreDistance::between(&profile_a, &profile_b);
println!("Distance: {:.4}", distance.distance);
println!("Is similar (< 0.5)? {}", distance.is_similar(0.5));

// Breakdown by feature
for (feature, diff) in &distance.breakdown {
    println!("  {}: {:.4}", feature, diff);
}
```

### `StyleCluster`

Groups agents by output similarity using single-linkage clustering.

```rust
use ternary_timbre::{TimbreProfile, StyleCluster};

let profiles = vec![
    TimbreProfile::from_features("agent-a", 0.5, 0.3, 0.2, 0.7),
    TimbreProfile::from_features("agent-b", 0.52, 0.31, 0.21, 0.69),
    TimbreProfile::from_features("agent-c", 2.0, 1.5, 1.8, 0.1),
];

let mut clusterer = StyleCluster::new(0.5);
clusterer.fit(&profiles);

// agent-a and agent-b should cluster together
assert_eq!(clusterer.find_cluster("agent-a"), clusterer.find_cluster("agent-b"));
// agent-c is in its own cluster
assert_ne!(clusterer.find_cluster("agent-a"), clusterer.find_cluster("agent-c"));
```

### `TimbreEvolution`

Tracks how an agent's output style changes over time — like watching a musician's style evolve across albums.

```rust
use ternary_timbre::{TimbreProfile, TimbreEvolution};

let mut evolution = TimbreEvolution::new("agent-x");

evolution.add_snapshot(TimbreProfile::from_features("agent-x", 0.5, 0.3, 0.2, 0.7));
evolution.add_snapshot(TimbreProfile::from_features("agent-x", 0.6, 0.4, 0.3, 0.65));
evolution.add_snapshot(TimbreProfile::from_features("agent-x", 0.9, 0.7, 0.5, 0.5));

println!("Total drift: {:.4}", evolution.total_drift());

let shifts = evolution.detect_shifts(0.5);
println!("Sudden shifts at snapshots: {:?}", shifts);

let trends = evolution.trends();
for (feature, direction) in trends {
    println!("{}: {:?}", feature, direction);
}
```

## Use Cases

- **Agent Selection**: Pick the right agent for a task based on style compatibility
- **Quality Monitoring**: Detect when an agent's output quality drifts
- **Team Composition**: Build diverse agent teams with complementary styles
- **A/B Testing**: Compare agent variants by their output characteristics
- **Anomaly Detection**: Flag agents whose style suddenly changes

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-timbre = "0.1.0"
```

## License

MIT
