# NaVQA Dataset: Comprehensive Analysis

## Dataset Overview

The Navigation Video Question Answering (NaVQA) dataset represents a significant contribution to the field of embodied AI, specifically designed to evaluate long-horizon reasoning capabilities in robot navigation systems. Unlike existing video QA datasets, NaVQA focuses on robot-centric scenarios with extended time horizons and includes spatial, temporal, and descriptive questions that are directly relevant to navigation tasks.

## Dataset Construction

### Source Data

NaVQA is built upon the CODa (Campus Object Dataset) robot navigation dataset, which features:
- Long-horizon navigation sequences in urban environments
- Indoor and outdoor settings on a university campus
- Data collected using a Clearpath Husky robot
- Various environmental conditions (sunny, cloudy, low-light, rainy)
- Multiple sensor streams (LIDAR, GPS, cameras)

From the 23 sequences in CODa, 7 were selected for NaVQA annotation, with each sequence ranging from 15 to 30 minutes in length.

### Sampling Strategy

To evaluate the impact of varying trajectory lengths on question-answering performance, the sequences were divided into three categories:
1. **Short** - Less than 2 minutes
2. **Medium** - Between 2 and 7 minutes
3. **Long** - Greater than 7 minutes

For each of the 7 selected sequences, 10 segments were sampled from each length category, resulting in 30 questions per sequence and a total of 210 questions across the dataset.

### Annotation Process

Five robot experts were recruited to annotate the dataset, ensuring that the questions and answers accurately reflected robot perception capabilities. The annotation process focused on creating questions that would test:
- Spatial understanding
- Object detection
- Sign reading
- Dynamic event understanding
- Contextual reasoning

## Question Types and Distribution

NaVQA encompasses five distinct question types, each requiring different forms of reasoning and output:

### 1. Binary Yes/No Questions (32%)
- Example: "Was the sidewalk busy today?"
- Output: Boolean (yes/no)
- Purpose: Test basic recognition and recall

### 2. Point-in-Time Questions (14%)
- Example: "When did you see the boxes fall?"
- Output: Relative time (e.g., "15 minutes ago")
- Purpose: Test temporal localization of events

### 3. Duration Questions (4%)
- Example: "How long were you inside the building for?"
- Output: Time span (e.g., "10 minutes")
- Purpose: Test temporal span understanding

### 4. Spatial Position Questions (34%)
- Example: "Where is the closest bathroom?"
- Output: Coordinate triplet (x, y, z)
- Purpose: Test spatial reasoning and localization

### 5. Descriptive Text Questions (16%)
- Example: "What side of the street are you driving on?"
- Output: Free-form text
- Purpose: Test general understanding and reasoning

## Evaluation Metrics

NaVQA employs specialized metrics for each question type:

### Spatial Metrics
- **L2 Distance Error** - Euclidean distance between predicted and ground truth coordinates
- **Correctness Threshold** - A spatial answer is considered correct if within 15 meters of the ground truth

### Temporal Metrics
- **L1 Temporal Error** - Absolute difference between predicted and ground truth times
- **Correctness Threshold** - A temporal answer is considered correct if within 2 minutes of the ground truth

### Descriptive Metrics
- **Binary Accuracy** - For yes/no and textual answers
- **LLM-Based Evaluation** - For more complex textual answers, an LLM evaluates correctness

### Overall Correctness
To provide a unified metric across question types, the thresholded correctness scores are combined into an "Overall Correctness" metric, which represents the proportion of questions answered correctly across all types.

## Dataset Statistics

### Question Distribution by Length Category
- Short videos: 70 questions (33.3%)
- Medium videos: 70 questions (33.3%)
- Long videos: 70 questions (33.3%)

### Question Distribution by Type
- Binary yes/no questions: 67 questions (32%)
- Point-in-time questions: 29 questions (14%)
- Duration questions: 8 questions (4%)
- Spatial position questions: 71 questions (34%)
- Descriptive text questions: 35 questions (16%)

### Environmental Coverage
- Indoor environments: ~30% of questions
- Outdoor environments: ~70% of questions
- Morning scenarios: ~25% of questions
- Afternoon scenarios: ~50% of questions
- Evening/low-light scenarios: ~25% of questions

## Unique Challenges

NaVQA presents several unique challenges that differentiate it from existing datasets:

### 1. Long-Horizon Reasoning
Models must reason over extended time periods (up to 20+ minutes), requiring efficient memory management and retrieval.

### 2. Robot-Centric Perception
Questions are framed from the robot's perspective, focusing on what the robot has observed during its deployment.

### 3. Actionable Outputs
Many questions require outputs that can be directly used for navigation (e.g., coordinates), not just descriptive answers.

### 4. Multi-Modal Integration
Successful models must integrate visual information with spatial and temporal data.

### 5. Dynamic Environments
Unlike static scene datasets, NaVQA includes questions about dynamic events and changing environments.

## Benchmark Results

The NaVQA paper presents baseline results using several approaches:

### ReMEmbR with Various LLMs
- GPT-4o: Best overall performance, especially on long videos
- Codestral: Moderate performance, struggles with arithmetic reasoning
- Command-R: Similar to Codestral, good at function calling but weaker at spatial/temporal reasoning
- Llama3.1-8b: Fastest but least accurate, especially on complex questions

### Comparison Methods
- LLM with Captions: Processes all captions at once, performs well on short videos but struggles with longer ones
- Multi-Frame VLM: Processes frames directly, strong on short videos but cannot handle medium/long videos due to context limitations

### Key Performance Insights
- Descriptive question accuracy decreases with video length for most models except ReMEmbR with GPT-4o
- Positional error increases with video length across all models
- Temporal error shows similar trends, with ReMEmbR maintaining the lowest error rates

## Research Applications

NaVQA enables research in several key areas:

### 1. Memory Systems for Robotics
Testing different memory architectures for long-horizon robot deployment.

### 2. Multi-Modal Reasoning
Evaluating models that combine visual, spatial, and temporal information.

### 3. Navigational Intelligence
Developing systems that can answer questions and generate navigational goals.

### 4. Scalable Robot Perception
Creating models that can efficiently process and reason over extended robot experiences.

## Dataset Access

The NaVQA dataset, along with evaluation code and baseline implementations, is available at the project website: [https://nvidia-ai-iot.github.io/remembr](https://nvidia-ai-iot.github.io/remembr)

## Links to Related Documentation

- [ReMEmbR Architecture](remembr_architecture.md) - Details on the system architecture
- [Experimental Results](experimental_results.md) - Detailed performance analysis
- [Real-World Applications](real_world_applications.md) - Case studies of deployment
- [Technical Challenges](technical_challenges.md) - Ongoing research challenges 