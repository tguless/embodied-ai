# Real-World Applications: ReMEmbR Deployment Case Studies

## Introduction

While the NaVQA dataset provides a standardized benchmark for evaluating ReMEmbR's capabilities, real-world deployment offers crucial insights into the system's practical utility and limitations. This document details the deployment of ReMEmbR on physical robots, the challenges encountered, and the observed performance in various real-world scenarios.

## Deployment Platform

### Hardware Configuration

ReMEmbR was deployed on a Nova Carter robot with the following specifications:

- **Compute Platform**: NVIDIA Jetson Orin 32GB
- **Sensors**:
  - RGB cameras for visual perception
  - 3D LiDAR for mapping and localization
  - IMU for motion tracking
  - Wheel encoders for odometry
- **Locomotion**: Differential drive system
- **Power**: Battery-powered with approximately 4-hour operation time

### Software Stack

The deployment utilized a comprehensive software stack:

- **Operating System**: Ubuntu 20.04 with ROS2 Foxy
- **Navigation**: ROS2's Nav2 stack with AMCL for localization
- **Mapping**: Pre-mapped environments using 3D LiDAR
- **Perception**:
  - Quantized VILA-3b for video captioning
  - Whisper ASR (optimized for Jetson) for voice interaction
- **Memory System**:
  - Vector database for memory storage
  - LLM backend for querying (various options)

## Deployment Environment

The system was tested in a large office space with the following characteristics:

- **Area**: Approximately 500 square meters
- **Layout**: Open office areas, corridors, meeting rooms, and common spaces
- **Features**: Furniture, equipment, signage, and dynamic elements (people, movable objects)
- **Lighting**: Mixed lighting conditions (natural and artificial)

## Deployment Procedure

### Memory Building Phase

1. The robot was driven around the environment for approximately 25 minutes
2. During this time, it continuously:
   - Captured visual data from cameras
   - Recorded position data from localization
   - Generated and embedded captions for video segments
   - Stored embeddings, positions, and timestamps in the vector database

### Querying Phase

After building the memory, users interacted with the robot through:
1. Voice commands processed by the Whisper ASR system
2. Text input via a connected interface
3. Pre-programmed queries for systematic testing

## Case Studies

### Case Study 1: Object Localization

**Query**: "Where can I get some chips?"

**System Response**:
1. The LLM agent formulated text retrieval queries for "chips", "snacks", and related terms
2. Retrieved memories included a cafeteria shelf containing snack items
3. The system provided coordinates to this location
4. The robot successfully navigated to the cafeteria shelf containing chips

**Key Observations**:
- The system correctly distinguished between different food locations (cafeteria vs. vending machines)
- The positional accuracy was sufficient for practical navigation
- The robot correctly identified the specific shelf rather than just the general cafeteria area

### Case Study 2: Semantic Goal Navigation

**Query**: "Take me somewhere with a nice view"

**System Response**:
1. The LLM agent formulated queries for aesthetic concepts like "view", "windows", "plants", etc.
2. Function calls searched for tall glass windows, plants, and open spaces
3. The system identified a lobby area with large glass windows and greenery
4. The robot navigated to this location successfully

**Key Observations**:
- The system demonstrated understanding of abstract concepts like "nice view"
- It correctly associated environmental features (windows, plants, open spaces) with the concept
- The retrieval process effectively identified memories with relevant aesthetic qualities

### Case Study 3: Ambiguous Object Identification

**Query**: "Take me to the soda machine"

**System Response**:
1. The LLM agent searched for "soda machine", "vending machine", etc.
2. Retrieved memories included a water fountain captioned as a "silver machine"
3. The system navigated to the water fountain instead of a soda machine

**Key Observations**:
- The quantized 3B captioning model misidentified the water fountain
- This highlights a limitation of using smaller, quantized models for captioning
- The error propagated through the retrieval and reasoning process

### Case Study 4: Temporal Reasoning

**Query**: "When was the last time you saw people in the break room?"

**System Response**:
1. The LLM agent searched for "break room", "people", etc.
2. Retrieved memories with timestamps were analyzed
3. The system provided a relative time answer ("about 15 minutes ago")

**Key Observations**:
- The system successfully integrated temporal information with spatial and visual data
- Relative time expressions were correctly calculated based on current time
- The answer was verified to be accurate within the 2-minute threshold

### Case Study 5: Multi-Step Reasoning

**Query**: "Which meeting room had the most people today?"

**System Response**:
1. First retrieval: Identified all meeting rooms in the environment
2. Second retrieval: For each room, searched for instances of people
3. Third retrieval: Refined search to focus on rooms with larger groups
4. The system identified "Conference Room B" as having the most people

**Key Observations**:
- The iterative retrieval process was essential for this complex query
- The system successfully counted and compared across different spatial locations
- The answer required integration of spatial, visual, and quantitative reasoning

## Performance Analysis

### Response Time

In real-world deployment, query response times were consistent with laboratory results:
- Average response time: 27.3 seconds
- Range: 18-42 seconds depending on query complexity
- No significant correlation between response time and memory size

### Accuracy

Based on manual verification of 50 test queries:
- Spatial questions: 82% accuracy (within 15m threshold)
- Temporal questions: 78% accuracy (within 2min threshold)
- Descriptive questions: 74% accuracy

### User Experience

Feedback from 12 test users highlighted several aspects:
- Response time was generally acceptable but at the upper limit of tolerance
- Voice interaction was preferred over text input
- Users appreciated the ability to ask natural language questions
- Some frustration occurred with misunderstood queries or inaccurate responses

## Technical Challenges

Several challenges were encountered during real-world deployment:

### 1. Captioning Quality

The quantized VILA-3b model occasionally produced inaccurate or vague captions, leading to retrieval errors. Examples included:
- Misidentifying objects (water fountain as "silver machine")
- Omitting important details in complex scenes
- Using generic descriptions for specialized equipment

### 2. Resource Constraints

The Jetson Orin 32GB provided sufficient compute for the memory building phase but with limitations:
- Caption generation was limited to 2 FPS to maintain real-time performance
- Vector database operations occasionally caused memory pressure
- Running larger LLM models locally was not feasible

### 3. Localization Precision

While ROS2's Nav2 with AMCL provided adequate localization, some challenges emerged:
- Positional drift in large open areas with few features
- Temporary localization uncertainty when moving through crowded areas
- Slight discrepancies between remembered positions and current map

### 4. Voice Interaction

The Whisper ASR system performed well but with some limitations:
- Difficulty with specialized terminology or proper nouns
- Reduced accuracy in noisy environments
- Occasional latency in processing longer queries

## Integration with Navigation Systems

A key aspect of the deployment was the integration with the robot's navigation stack:

1. For spatial queries, ReMEmbR provided (x,y) coordinates in the map frame
2. These coordinates were passed to Nav2 as goal positions
3. The navigation stack handled path planning and execution
4. For descriptive or temporal queries, the robot remained stationary while responding

This integration enabled seamless transitions from query responses to navigation actions, allowing the robot to not just answer questions but act on them.

## Deployment Insights

### Key Successes

1. **Consistent Performance**: The system maintained reliable performance over extended operation periods (4+ hours)

2. **Scalable Memory**: The vector database efficiently handled growing memory size without performance degradation

3. **Natural Interaction**: Users could interact with the robot using natural language without specialized commands

4. **Actionable Intelligence**: The system successfully converted queries into navigational goals

### Areas for Improvement

1. **Captioning Quality**: Higher quality captions would improve retrieval accuracy, possibly through:
   - Using larger, less quantized models
   - Fine-tuning captioning models on robot-specific data
   - Incorporating multi-view captioning for better scene understanding

2. **Response Time**: Reducing the average response time below 20 seconds would improve user experience

3. **Error Recovery**: Better handling of misunderstood queries or failed retrievals

4. **Memory Management**: More sophisticated memory consolidation to reduce redundancy in long-term storage

## Future Deployment Directions

Based on the initial deployment experience, several promising directions for future deployments have been identified:

### 1. Multi-Robot Memory Sharing

Extending ReMEmbR to share memories across multiple robots operating in the same environment:
- Distributed memory building across robot fleet
- Shared vector database with robot-specific annotations
- Cross-robot query capabilities ("Has any robot seen...")

### 2. Long-Term Memory Evolution

Implementing mechanisms for long-term memory management:
- Importance-based memory retention
- Consolidation of repetitive observations
- Forgetting strategies for outdated information

### 3. Multi-Modal Memory

Incorporating additional sensor modalities beyond vision:
- Audio memory for sound events
- Tactile feedback for surface properties
- Temperature and other environmental sensors

### 4. User-Specific Adaptation

Personalizing the system to individual users:
- Learning user preferences and frequently asked questions
- Adapting response style and detail level
- Building user-specific spatial references

## Conclusion

The real-world deployment of ReMEmbR on the Nova Carter robot demonstrates the practical viability of retrieval-augmented memory systems for robot navigation. Despite some limitations in captioning quality and response time, the system successfully enabled natural language interaction with the robot's spatial and temporal memory.

The deployment validated key aspects of the ReMEmbR architecture, particularly the scalability of the vector database approach and the effectiveness of the iterative retrieval mechanism. It also highlighted areas for improvement, especially in caption quality and response time optimization.

As robot deployments continue to extend in duration and complexity, systems like ReMEmbR will become increasingly valuable for maintaining and reasoning over long-horizon memories of robot experiences.

## Links to Related Documentation

- [ReMEmbR Architecture](remembr_architecture.md) - Details on the system architecture
- [NaVQA Dataset Analysis](navqa_dataset.md) - Information about the evaluation dataset
- [Experimental Results](experimental_results.md) - Performance metrics and comparisons
- [Technical Challenges](technical_challenges.md) - Ongoing research challenges 