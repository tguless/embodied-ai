# ReMEmbR System Diagrams

This document provides visual representations of the ReMEmbR system architecture, components, and related datasets to complement the textual documentation.

## ReMEmbR Architecture Overview

The following diagram illustrates the two main phases of the ReMEmbR system: the Memory Building Phase and the Querying Phase. It shows the data flow from robot sensors through to answer generation.

```mermaid
graph TD
    subgraph "Memory Building Phase"
        A["Robot Sensors"] --> B["Image Data"]
        A --> C["Position Data"]
        A --> D["Timestamp Data"]
        B --> E["Video Segmentation<br/>3-second segments at 2 FPS"]
        E --> F["VILA Video Captioning Model"]
        F --> G["Caption Generation"]
        G --> H["Text Embedding<br/>mxbai-embed-large-v1"]
        H --> I["Vector Database Storage"]
        C --> I
        D --> I
    end
    
    subgraph "Querying Phase"
        J["User Question"] --> K["LLM Agent<br/>GPT-4o/Command-R/Llama3.1"]
        K --> L{"Sufficient<br/>Context?"}
        L -->|No| M["Retrieval Functions"]
        M --> N["Text Retrieval<br/>fl(object)"]
        M --> O["Position Retrieval<br/>fp((x,y,z))"]
        M --> P["Time Retrieval<br/>ft(HH:MM:SS)"]
        N --> Q["Retrieved Memories"]
        O --> Q
        P --> Q
        Q --> K
        L -->|Yes| R["Answer Generation"]
        R --> S["Structured JSON Output"]
        S --> T["Text Answer"]
        S --> U["Position Coordinates"]
        S --> V["Temporal Information"]
    end
    
    I -.-> M
```

### Key Components:

1. **Memory Building Phase**:
   - Processes robot sensor data (images, positions, timestamps)
   - Segments video into 3-second chunks at 2 FPS
   - Uses VILA model for video captioning
   - Embeds captions using mxbai-embed-large-v1
   - Stores embeddings along with position and timestamp data in a vector database

2. **Querying Phase**:
   - Takes user questions as input
   - Uses an LLM agent to process the question
   - Iteratively retrieves relevant memories using specialized functions
   - Generates structured answers in JSON format with text, position, or temporal information

## System Integration Diagram

This diagram shows how ReMEmbR integrates with hardware and software components in a real-world deployment.

```mermaid
flowchart TD
    subgraph "Hardware Layer"
        A["Nova Carter Robot"] --> B["Jetson Orin 32GB"]
        A --> C["RGB Cameras"]
        A --> D["3D LiDAR"]
        A --> E["Wheel Encoders"]
    end
    
    subgraph "Software Stack"
        F["ROS2 Nav2"] --> G["AMCL Localization"]
        H["Whisper ASR"] --> I["Voice Commands"]
        J["VILA-3b"] --> K["Video Captioning"]
        L["Vector Database"] --> M["Memory Storage"]
        N["LLM Backend"] --> O["Query Processing"]
    end
    
    subgraph "ReMEmbR System"
        P["Memory Building Phase"] --> Q["Video Processing"]
        P --> R["Position Tracking"]
        P --> S["Timestamp Recording"]
        T["Querying Phase"] --> U["Query Understanding"]
        T --> V["Memory Retrieval"]
        T --> W["Answer Generation"]
    end
    
    B --> F
    B --> H
    B --> J
    B --> L
    B --> N
    C --> J
    D --> G
    E --> G
    K --> P
    G --> R
    M --> V
    O --> U
    O --> W
```

### Key Integration Points:

1. **Hardware Layer**:
   - Nova Carter robot provides the physical platform
   - Jetson Orin 32GB serves as the compute platform
   - Sensors include RGB cameras, 3D LiDAR, and wheel encoders

2. **Software Stack**:
   - ROS2 Nav2 with AMCL handles navigation and localization
   - Whisper ASR processes voice commands
   - VILA-3b generates video captions
   - Vector database stores and retrieves memories
   - LLM backend processes queries

3. **ReMEmbR System**:
   - Memory Building Phase processes sensor data
   - Querying Phase handles user interactions and memory retrieval

## NaVQA Dataset Structure

This diagram illustrates the structure of the NaVQA dataset used to evaluate ReMEmbR, including question types, output formats, and evaluation metrics.

```mermaid
flowchart TD
    A["NaVQA Dataset<br/>210 Questions"] --> B["Short Videos<br/><2 minutes<br/>70 questions"]
    A --> C["Medium Videos<br/>2-7 minutes<br/>70 questions"]
    A --> D["Long Videos<br/>>7 minutes<br/>70 questions"]
    
    subgraph "Question Types"
        E["Binary Yes/No<br/>32%"]
        F["Point-in-Time<br/>14%"]
        G["Duration<br/>4%"]
        H["Spatial Position<br/>34%"]
        I["Descriptive Text<br/>16%"]
    end
    
    subgraph "Output Types"
        J["Boolean"]
        K["Relative Time<br/>(e.g., '15 minutes ago')"]
        L["Time Span<br/>(e.g., '10 minutes')"]
        M["Coordinates<br/>(x, y, z)"]
        N["Free-form Text"]
    end
    
    E --> J
    F --> K
    G --> L
    H --> M
    I --> N
    
    subgraph "Evaluation Metrics"
        O["Binary Accuracy"]
        P["L1 Temporal Error"]
        Q["L2 Distance Error"]
        R["Overall Correctness"]
    end
    
    J --> O
    N --> O
    K --> P
    L --> P
    M --> Q
    O --> R
    P --> R
    Q --> R
```

### Key Dataset Features:

1. **Video Categories**:
   - Short videos (< 2 minutes): 70 questions
   - Medium videos (2-7 minutes): 70 questions
   - Long videos (> 7 minutes): 70 questions

2. **Question Types**:
   - Binary Yes/No questions (32%)
   - Point-in-Time questions (14%)
   - Duration questions (4%)
   - Spatial Position questions (34%)
   - Descriptive Text questions (16%)

3. **Output Types**:
   - Boolean values for yes/no questions
   - Relative time expressions for point-in-time questions
   - Time spans for duration questions
   - Coordinates for spatial questions
   - Free-form text for descriptive questions

4. **Evaluation Metrics**:
   - Binary accuracy for yes/no and descriptive questions
   - L1 temporal error for time-related questions
   - L2 distance error for spatial questions
   - Overall correctness as a unified metric

## Links to Related Documentation

- [ReMEmbR Architecture](remembr_architecture.md) - Detailed technical analysis
- [NaVQA Dataset Analysis](navqa_dataset.md) - Comprehensive dataset information
- [Experimental Results](experimental_results.md) - Performance evaluation
- [Real-World Applications](real_world_applications.md) - Deployment case studies
- [Technical Challenges](technical_challenges.md) - Research challenges and future directions 