# ReMEmbR Architecture: In-Depth Technical Analysis

## Introduction

ReMEmbR (Retrieval-augmented Memory for Embodied Robots) represents a significant advancement in robot memory systems, specifically designed to address the challenges of long-horizon navigation and reasoning. This document provides a detailed technical analysis of the ReMEmbR architecture, its components, and the underlying mechanisms that enable effective long-horizon spatio-temporal reasoning.

## System Overview

ReMEmbR is fundamentally structured around two core phases:
1. **Memory Building Phase** - The continuous accumulation and structuring of robot experiences
2. **Querying Phase** - The retrieval and reasoning over accumulated memories to answer questions

This dual-phase approach allows ReMEmbR to efficiently handle unbounded history lengths, a key limitation in previous approaches that relied on fixed context windows or transformer-based methods.

## Mathematical Formulation

The core challenge addressed by ReMEmbR is computing the probability of an answer A given a history H and question Q:

```
p(A|H₁:ₖ, Q)
```

Where H₁:ₖ represents the robot's history over K minutes of deployment.

Due to computational constraints, ReMEmbR reformulates this as:

```
p(A|H₁:ₖ, Q) = p(A|R*, Q) ≈ p(A|R, Q)
```

Where:
- R* is the optimal subset of history needed to answer the question
- R is an approximation of R* sampled using function F: V → R
- V is the memory representation

The goal is to minimize the size of R while ensuring consistent answers:

```
R* = argmin |R|
      R
s.t. argmax p(A|R, Q) = argmax p(A'|H, Q)
      A                   A'
```

## Memory Building Phase

### Data Collection and Processing

During deployment, the robot continuously collects:
1. **Image data (Hᴵ)** - Visual information from onboard cameras
2. **Position data (Hᴾ)** - Localization information (x,y,z coordinates)
3. **Timestamp data (Hᵀ)** - Temporal markers for all observations

### Video Segmentation and Captioning

ReMEmbR processes the continuous video stream in t-second segments:
- Each segment Hᴵᵢ:ᵢ₊ₜ is processed by the VILA video captioning model
- VILA 1.5-13b is the default model, though smaller variants (8b, 3b) can be used with minimal performance degradation
- The optimal segment length is 3 seconds at 2 FPS (6 frames)

### Caption Embedding

For each video segment caption Lᵢ:ᵢ₊ₜ:
- Text embedding function E generates vector representations
- The default embedding model is mxbai-embed-large-v1
- The embedding captures semantic content of what the robot observes

### Vector Database Storage

The vector database V stores:
- Caption embeddings E(Lᴵᵢ:ᵢ₊ₜ)
- Position data Hᴾᵢ:ᵢ₊ₜ
- Timestamp data Hᵀᵢ:ᵢ₊ₜ

This database is optimized for efficient retrieval using quantized approximate nearest neighbor methods, allowing it to scale to millions of entries.

## Querying Phase

### LLM-Agent as State Machine

The querying phase employs an LLM-agent that functions as a state machine:
1. **Initialization** - The agent receives a question Q and empty context R₀
2. **Retrieval** - The agent formulates queries to retrieve relevant memories
3. **Assessment** - The agent evaluates if sufficient context exists to answer the question
4. **Iteration** - If needed, the agent returns to the retrieval step
5. **Response** - Once sufficient context is gathered, the agent formulates an answer

### Retrieval Functions

The LLM-agent can call three specialized retrieval functions:
1. **Text retrieval**: `fl(object) → m memories`
   - Searches for memories containing specific objects or concepts
   - Example: `fl("bathroom") → [memory₁, memory₂, ..., memoryₘ]`

2. **Position retrieval**: `fp((x, y, z)) → m memories`
   - Retrieves memories associated with specific spatial coordinates
   - Example: `fp((42.3, -71.1, 0)) → [memory₁, memory₂, ..., memoryₘ]`

3. **Time retrieval**: `ft("HH:MM:SS") → m memories`
   - Fetches memories from specific timestamps
   - Example: `ft("14:35:22") → [memory₁, memory₂, ..., memoryₘ]`

### Iterative Query Refinement

For each iteration i:
1. The LLM generates a function call f and query q based on current context R₀:ᵢ and question Q
2. The function retrieves m new memories: Rᵢ:ᵢ₊ₘ = f(q), where q = LLM(R₀:ᵢ, Q)
3. The LLM can formulate up to k queries, retrieving k×m memories in total
4. After retrieval, the LLM assesses if the question can be answered with the updated context

### Answer Generation

Once sufficient context is gathered:
1. The LLM summarizes relevant information from all retrieved memories
2. It generates a structured answer in JSON format with appropriate fields:
   - `text` for descriptive answers
   - `position` for spatial coordinates
   - `time` or `duration` for temporal answers

## Implementation Details

### Hardware Requirements

For the full system:
- Memory building phase: Jetson Orin 32GB (for real-time deployment)
- Querying phase: Compatible with various LLM backends:
  - Cloud-based: GPT-4o, NVIDIA NIM APIs
  - Local large models: Command-R, Codestral
  - On-device: Llama3.1-8b or similar function-calling LLMs

### Software Stack

For robot deployment:
- ROS2's Nav2 stack with AMCL for localization
- Whisper ASR for voice interaction
- VILA-3b (quantized) for video captioning
- Vector database (FAISS or similar) for memory storage

### Performance Considerations

- Caption generation: 2 FPS is optimal (6 frames per 3-second segment)
- Lower frame rates (0.5 FPS) lead to information loss and reduced performance
- Query latency: ~25 seconds per question (relatively constant regardless of history length)
- Memory footprint: Scales linearly with deployment duration

## Integration with Navigation Systems

ReMEmbR is designed to integrate with existing robot navigation frameworks:
1. The memory building phase runs continuously during robot operation
2. When a query is received, the system retrieves relevant memories
3. For navigational queries, the system outputs actionable coordinates
4. These coordinates can be passed directly to the robot's navigation stack

## Links to Related Documentation

- [NaVQA Dataset Analysis](navqa_dataset.md) - Details on the evaluation dataset
- [Experimental Results](experimental_results.md) - Performance metrics and comparisons
- [Real-World Applications](real_world_applications.md) - Deployment case studies
- [Technical Challenges](technical_challenges.md) - Ongoing research challenges 