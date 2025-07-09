# Experimental Results: Detailed Analysis of ReMEmbR Performance

## Experimental Setup

### Evaluation Dataset

All experiments were conducted using the NaVQA dataset, which consists of 210 questions across three video length categories:
- **Short** - Less than 2 minutes
- **Medium** - Between 2 and 7 minutes
- **Long** - Greater than 7 minutes

### Evaluated Methods

The experiments compared several configurations of ReMEmbR against baseline approaches:

#### ReMEmbR Variants
- **ReMEmbR with GPT-4o** - Primary configuration using OpenAI's GPT-4o as the LLM backend
- **ReMEmbR with Codestral** - Using the open-source Codestral model (22B parameters)
- **ReMEmbR with Command-R** - Using Cohere's Command-R model
- **ReMEmbR with Llama3.1-8b** - Using a smaller 8B parameter model

#### Baseline Methods
- **LLM with Caption** - Providing all video captions directly to GPT-4o without retrieval
- **Multi-Frame VLM** - Processing frames sampled at 2 FPS directly with GPT-4o

### Evaluation Metrics

Four primary metrics were used to evaluate performance:

1. **Descriptive Question Accuracy** - Binary accuracy for yes/no and textual answers
2. **Positional Error** - L2 distance (in meters) between predicted and ground truth coordinates
3. **Temporal Error** - L1 difference (in seconds) between predicted and ground truth times
4. **Overall Correctness** - Proportion of questions answered correctly using thresholds:
   - Spatial: Within 15 meters of ground truth
   - Temporal: Within 2 minutes of ground truth
   - Descriptive: Binary accuracy

## Core Results

### Performance Across Video Lengths

The table below summarizes the performance of different methods across the three video length categories:

| Method | Descriptive Question Accuracy (Short/Medium/Long) | Positional Error in meters (Short/Medium/Long) | Temporal Error in seconds (Short/Medium/Long) |
|--------|--------------------------------------------------|-----------------------------------------------|-----------------------------------------------|
| ReMEmbR (GPT-4o) | 0.62±0.5 / 0.58±0.5 / 0.65±0.5 | 5.1±11.9 / 27.5±26.8 / 46.25±59.6 | 0.3±0.1 / 1.8±2.0 / 3.6±5.9 |
| ReMEmbR (Codestral) | 0.25±0.4 / 0.24±0.4 / 0.11±0.3 | 151.3±109.7 / 189.0±109.6 / 212.4±121.3 | 4.8±5.6 / 8.4±6.8 / 14.8±7.5 |
| ReMEmbR (Command-R) | 0.36±0.5 / 0.32±0.5 / 0.14±0.3 | 158.7±129.6 / 172.2±119.4 / 188.7±107.1 | 4.5±17.3 / 14.3±6.7 / 15.3±11.7 |
| ReMEmbR (Llama3.1-8b) | 0.31±0.5 / 0.33±0.5 / 0.21±0.4 | 159.9±123.2 / 151.2±121.1 / 165.3±115.1 | 9.5±27.5 / 7.9±16.3 / 18.7±10.8 |
| LLM with Caption | 0.57±0.5 / 0.66±0.5 / 0.55±0.5 | 5.1±8.2 / 33.3±47.3 / 56.0±61.7 | 0.5±0.5 / 1.9±2.2 / 8.0±6.7 |
| Multi-Frame VLM | 0.55±0.5 / ✗ / ✗ | 7.5±11.4 / ✗ / ✗ | 0.5±2.2 / ✗ / ✗ |

Note: ✗ indicates that the method could not process videos of this length due to context limitations.

### Overall Correctness Analysis

The overall correctness metric provides a unified view of performance across all question types:

| Method | Overall Correctness (Short/Medium/Long) |
|--------|----------------------------------------|
| ReMEmbR (GPT-4o) | 0.72±0.5 / 0.67±0.5 / 0.54±0.5 |
| ReMEmbR (1 call only) | 0.56±0.5 / 0.48±0.4 / 0.50±0.5 |
| ReMEmbR (12-sec captions) | 0.61±0.5 / 0.50±0.5 / 0.38±0.5 |
| ReMEmbR (VILA1.5-8b) | 0.58±0.5 / 0.52±0.5 / 0.54±0.5 |
| ReMEmbR (VILA1.5-3b) | 0.60±0.5 / 0.58±0.5 / 0.50±0.5 |

### Performance Over Time

Analysis of overall correctness across different time ranges shows that ReMEmbR maintains higher performance as video length increases compared to baseline methods:

| Time Range | ReMEmbR (GPT-4o) | LLM with Caption | Multi-Frame VLM |
|------------|------------------|------------------|-----------------|
| 0-5 minutes | 0.71 | 0.73 | 0.70 |
| 5-10 minutes | 0.68 | 0.62 | N/A |
| 10-15 minutes | 0.60 | 0.53 | N/A |
| 15-20+ minutes | 0.55 | 0.48 | N/A |

## Detailed Analysis

### Impact of LLM Choice

The experiments demonstrate a significant performance gap between different LLM backends:

1. **GPT-4o** consistently delivers the best performance across all metrics, with particularly strong results in:
   - Maintaining descriptive question accuracy even for long videos
   - Achieving low positional error (5.1m for short videos, 46.25m for long videos)
   - Maintaining low temporal error (0.3s for short videos, 3.6s for long videos)

2. **Open-source models** (Codestral, Command-R, Llama3.1-8b) show significantly weaker performance:
   - Descriptive accuracy drops to 0.11-0.33 (vs. 0.58-0.65 for GPT-4o)
   - Positional error increases by 3-4x (151-212m vs. 5-46m)
   - Temporal error increases by 3-5x (4.8-18.7s vs. 0.3-3.6s)

The primary limitation of open-source models appears to be arithmetic reasoning required for spatial and temporal questions, rather than the function-calling capability itself.

### Ablation Studies

Several ablation studies provide insights into key components of ReMEmbR:

#### 1. Number of Retrieval Iterations

Limiting ReMEmbR to a single retrieval call (similar to traditional RAG) reduces overall correctness from 0.72/0.67/0.54 to 0.56/0.48/0.50 across short/medium/long videos. This demonstrates the importance of iterative retrieval for:
- Multi-step reasoning
- Recovery from unsuccessful initial retrievals
- Building comprehensive context for complex questions

#### 2. Caption Temporal Resolution

Reducing caption frequency from 3-second segments (2 FPS) to 12-second segments (0.5 FPS) decreases overall correctness from 0.72/0.67/0.54 to 0.61/0.50/0.38. This indicates that:
- Higher temporal resolution captures more detailed information
- Information loss from coarser sampling significantly impacts performance
- The effect is most pronounced on longer videos (29% drop for long vs. 15% for short)

#### 3. Captioning Model Size

Comparing different VILA model sizes reveals:
- VILA1.5-13b (default): 0.72/0.67/0.54 overall correctness
- VILA1.5-8b: 0.58/0.52/0.54 overall correctness
- VILA1.5-3b: 0.60/0.58/0.50 overall correctness

The minimal performance loss when using smaller models is significant for deployment scenarios where computational resources are limited.

### Latency Analysis

A critical advantage of ReMEmbR is its consistent query latency regardless of video length:

| Method | Short Video Latency | Medium Video Latency | Long Video Latency |
|--------|---------------------|----------------------|---------------------|
| ReMEmbR (GPT-4o) | ~25 seconds | ~25 seconds | ~25 seconds |
| ReMEmbR (Command-R/Codestral) | ~40 seconds | ~40 seconds | ~40 seconds |
| ReMEmbR (Llama3.1-8b) | ~15 seconds | ~15 seconds | ~15 seconds |
| Multi-Frame VLM | ~90 seconds | N/A | N/A |

This consistent latency is achieved because ReMEmbR only processes a small subset of relevant memories rather than the entire video history.

## Question Type Analysis

Breaking down performance by question type reveals different strengths and weaknesses:

### Spatial Questions

- ReMEmbR with GPT-4o achieves the lowest positional error across all video lengths
- Error increases with video length for all methods, but ReMEmbR shows the slowest degradation
- Open-source models struggle significantly with coordinate prediction

### Temporal Questions

- ReMEmbR with GPT-4o maintains low temporal error even for long videos
- Point-in-time questions are handled more accurately than duration questions
- Open-source models show particular weakness in duration estimation

### Descriptive Questions

- Binary yes/no questions are handled more accurately than free-form text questions
- ReMEmbR with GPT-4o maintains consistent performance across video lengths
- Open-source models show better relative performance on descriptive questions compared to spatial/temporal questions

## Comparative Analysis

### ReMEmbR vs. LLM with Caption

- For short videos, performance is comparable (0.72 vs. 0.73 overall correctness)
- For medium and long videos, ReMEmbR significantly outperforms (0.67 vs. 0.62 and 0.54 vs. 0.48)
- The gap widens as video length increases, demonstrating ReMEmbR's scalability

### ReMEmbR vs. Multi-Frame VLM

- For short videos, performance is comparable (0.72 vs. 0.70 overall correctness)
- Multi-Frame VLM cannot process medium or long videos due to context limitations
- ReMEmbR's retrieval-based approach enables scaling to arbitrary video lengths

## Conclusion

The experimental results demonstrate that:

1. ReMEmbR effectively enables long-horizon reasoning for robot navigation, outperforming baseline approaches on videos longer than 5 minutes.

2. The iterative retrieval mechanism is crucial for performance, allowing multi-step reasoning and recovery from unsuccessful initial retrievals.

3. While GPT-4o delivers the best performance, smaller models can be used with acceptable performance degradation, enabling deployment in resource-constrained environments.

4. The system maintains consistent query latency regardless of video length, making it practical for real-world deployment.

5. Higher temporal resolution in captioning significantly improves performance, especially for longer videos.

These findings validate ReMEmbR's approach to building and querying long-horizon spatio-temporal memory for robot navigation.

## Links to Related Documentation

- [ReMEmbR Architecture](remembr_architecture.md) - Details on the system architecture
- [NaVQA Dataset Analysis](navqa_dataset.md) - Information about the evaluation dataset
- [Real-World Applications](real_world_applications.md) - Case studies of deployment
- [Technical Challenges](technical_challenges.md) - Ongoing research challenges 