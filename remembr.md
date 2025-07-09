# ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation

## Overview
ReMEmbR (Retrieval-augmented Memory for Embodied Robots) is a system designed for long-horizon video question answering for robot navigation. It enables robots to build and reason over memories accumulated during extended deployments, allowing them to answer spatial, temporal, and descriptive questions about their experiences.

## Key Challenges Addressed
- Robots continuously operate for long periods, gathering extensive histories
- Traditional approaches can only handle short durations (1-2 minutes)
- Fixed context windows in LLMs cannot accommodate unbounded history lengths
- Need for actionable spatial and temporal information for navigation

## System Architecture
ReMEmbR consists of two main phases:

### 1. Memory Building Phase
- Processes robot sensor data including images, positions, and timestamps
- Uses VILA video captioning model to generate captions for temporal segments
- Embeds captions using text embedding models
- Stores vector representations in a vector database for efficient retrieval

### 2. Querying Phase
- Uses an LLM-agent to sample relevant memories from the database
- Implements three retrieval functions:
  - Text retrieval: `fl(object) → m memories`
  - Position retrieval: `fp((x, y, z)) → m memories`
  - Time retrieval: `ft("HH:MM:SS") → m memories`
- Iteratively retrieves memories until sufficient context is gathered
- Formats answers as structured outputs (text, position, time, or duration)

## NaVQA Dataset
The authors created the Navigation Video Question Answering (NaVQA) dataset to evaluate long-horizon reasoning:
- Built on the CODa robot navigation dataset
- 210 questions across 7 sequences of varying lengths
- Question types:
  - Spatial questions (34%): "Where is the closest bathroom?"
  - Temporal point-in-time questions (14%): "When did you see the boxes fall?"
  - Temporal duration questions (4%): "How long were you inside the building for?"
  - Binary yes/no questions (32%): "Was the sidewalk busy today?"
  - Descriptive text questions (16%): "What side of the street are you driving on?"

## Experimental Results
- ReMEmbR outperforms traditional LLM and VLM baselines on long-horizon tasks
- Maintains higher overall correctness as video length increases
- Processes queries with low latency (~25 seconds) regardless of video duration
- GPT-4o performs better than open-source LLMs (Codestral, Command-R, Llama3.1)
- Iterative function calls improve performance over single-retrieval approaches

## Real-World Deployment
- Successfully deployed on a Nova Carter robot
- Memory building phase ran on Jetson Orin 32GB
- Used quantized VILA-3b for captioning
- Integrated with ROS2's Nav2 stack and AMCL for localization
- Demonstrated ability to handle navigation-centric questions like:
  - "Where can I get some chips?"
  - "Take me somewhere with a nice view"

## Limitations and Future Work
- Current approach relies solely on video captioning
- Could benefit from integration with semantic maps, scene graphs, etc.
- System adds potentially repetitive information to vector database
- Efficient memory aggregation of pertinent information is an area for future research

## Project Resources
The dataset, code, videos, and other materials can be found at: https://nvidia-ai-iot.github.io/remembr 