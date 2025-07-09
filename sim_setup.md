# Simulation Setup Plan: ReMEmbR + NVIDIA Isaac Sim

This document outlines the detailed plan for setting up a simulation environment using NVIDIA Isaac Sim to develop and test the ReMEmbR (Retrieval-augmented Memory for Embodied Robots) system before deploying it on physical hardware.

## 1. Environment Setup (Week 1)

### 1.1. Hardware and Software Prerequisites

#### Hardware Requirements
- PC with NVIDIA 4xxxx series GPU (12GB VRAM)
- Minimum 32GB RAM
- 500GB+ SSD storage

#### Software Installation
- [ ] Install NVIDIA Game Ready Driver (minimum 537.58 for Windows)
- [ ] Install NVIDIA CUDA Toolkit 12.x
- [ ] Install Docker (required for MilvusDB)
- [ ] Install Conda for environment management

### 1.2. Isaac Sim Installation

- [ ] Download and install NVIDIA Omniverse Launcher
- [ ] Install Isaac Sim 2023.1.1 or later through Omniverse Launcher
- [ ] Run Isaac Sim Compatibility Checker to verify system requirements
- [ ] Test Isaac Sim with basic scene to ensure proper operation
- [ ] Install Isaac Sim Python environment

### 1.3. ReMEmbR Framework Setup

- [ ] Clone the ReMEmbR repository
  ```bash
  git clone https://github.com/NVIDIA-AI-IOT/remembr.git
  cd remembr
  ```
- [ ] Set up VILA (Vision-Language Assistant)
  ```bash
  mkdir deps
  cd deps
  git clone https://github.com/NVlabs/VILA.git
  cd ..
  ./vila_setup.sh remembr
  ```
- [ ] Install Ollama for LLM support
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- [ ] Install Python dependencies
  ```bash
  conda activate remembr
  python -m pip install -r requirements.txt
  ```
- [ ] Install MilvusDB for vector database
  ```bash
  curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o launch_milvus_container.sh
  bash launch_milvus_container.sh start
  ```

### 1.4. ROS 2 Installation and Setup

- [ ] Install ROS 2 Foxy (Ubuntu) or ROS 2 Foxy Windows binary (Windows)
- [ ] Install Isaac ROS packages
- [ ] Set up ROS 2 workspace
  ```bash
  mkdir -p ~/ros2_ws/src
  cd ~/ros2_ws
  colcon build
  ```
- [ ] Configure ROS 2 environment
  ```bash
  echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
  ```

## 2. Isaac Sim + ReMEmbR Integration (Week 2)

### 2.1. Isaac Sim ROS 2 Bridge Setup

- [ ] Install Isaac Sim ROS 2 bridge extension
- [ ] Test basic ROS 2 communication with Isaac Sim
  ```bash
  # Terminal 1
  ros2 topic list  # Verify ROS 2 is running
  
  # Terminal 2 (after launching Isaac Sim with ROS 2 bridge)
  ros2 topic echo /isaac_sim/clock  # Verify bridge is working
  ```

### 2.2. ReMEmbR Integration with Isaac Sim

- [ ] Create Python script to connect Isaac Sim camera data to ReMEmbR VLM
  ```python
  # Example pseudocode
  import omni.isaac.core
  from remembr.memory.memory import MemoryItem
  from remembr.memory.milvus_memory import MilvusMemory
  
  # Initialize memory
  memory = MilvusMemory("isaac_sim_collection", db_ip='127.0.0.1')
  
  # Function to capture camera data and create memory items
  def capture_camera_data(camera_handle, robot_state):
      image = camera_handle.get_rgba_data()
      # Process image with VLM to get caption
      caption = process_with_vlm(image)
      
      # Create memory item
      item = MemoryItem(
          caption=caption,
          time=simulation_time,
          position=robot_state.position,
          theta=robot_state.rotation[2]  # Assuming yaw
      )
      memory.insert(item)
  ```

- [ ] Create Isaac Sim extension for ReMEmbR integration
  - Create extension folder structure
  - Implement memory building pipeline in extension
  - Add UI components for query interface

### 2.3. Test Basic Integration

- [ ] Verify camera data can be processed by VLM
- [ ] Confirm memory items are correctly stored in MilvusDB
- [ ] Test basic querying functionality

## 3. Warehouse Environment Creation (Week 3)

### 3.1. Warehouse Scene Setup

- [ ] Create or import warehouse environment using Isaac Sim assets
  - Use Warehouse Creator Extension if available
  - Include shelving, pallets, boxes, and other typical warehouse items
- [ ] Set up proper lighting conditions
- [ ] Configure physics properties for objects

### 3.2. Robot Model Setup

- [ ] Import or create mobile robot model with:
  - RGB-D camera(s)
  - 2D LiDAR sensor
  - IMU sensor
  - Wheel encoders
- [ ] Configure robot articulations and physics properties
- [ ] Set up robot control interface

### 3.3. Sensor Configuration

- [ ] Configure RGB-D camera
  ```python
  # Example camera setup in Isaac Sim
  from omni.isaac.sensor import Camera
  
  camera = Camera(
      prim_path="/robot/camera",
      resolution=(640, 480),
      position=(0, 0, 0.5),
      rotation=(0, 0, 0)
  )
  ```
- [ ] Set up LiDAR sensor for navigation
- [ ] Configure IMU and wheel encoders
- [ ] Verify all sensors are publishing data to ROS 2 topics

## 4. Navigation and Data Collection System (Week 4)

### 4.1. Navigation Stack Setup

- [ ] Set up ROS 2 Navigation2 (Nav2) in Isaac Sim
- [ ] Configure SLAM for mapping
  ```bash
  # Example SLAM launch command
  ros2 launch isaac_ros_slam isaac_ros_slam_launch.py
  ```
- [ ] Create map of warehouse environment
- [ ] Test basic navigation capabilities

### 4.2. Data Collection Pipeline

- [ ] Create automated data collection system
  ```python
  # Pseudocode for data collection
  def collect_data():
      # Move robot to different locations
      for waypoint in waypoints:
          navigate_to(waypoint)
          # Collect sensor data at each waypoint
          capture_camera_data(camera, robot_state)
          # Wait for processing
          time.sleep(1.0)
  ```
- [ ] Implement waypoint navigation for systematic data collection
- [ ] Set up data storage and organization

### 4.3. Memory Building Pipeline

- [ ] Integrate VLM for image captioning
  ```python
  from remembr.vision.vila_vision import VILAVision
  
  # Initialize VLM
  vlm = VILAVision()
  
  def process_with_vlm(image):
      # Generate caption from image
      caption = vlm.generate_caption(image)
      return caption
  ```
- [ ] Configure temporal and spatial information extraction
- [ ] Implement memory storage system with MilvusDB
- [ ] Test end-to-end memory building pipeline

## 5. ReMEmbR Agent Implementation (Week 5)

### 5.1. Agent Setup

- [ ] Initialize ReMEmbR agent with appropriate LLM
  ```python
  from remembr.agents.remembr_agent import ReMEmbRAgent
  
  # Create agent
  agent = ReMEmbRAgent(llm_type='command-r')
  
  # Connect to memory
  agent.set_memory(memory)
  ```
- [ ] Configure agent parameters
- [ ] Set up query interface

### 5.2. Function Calling System

- [ ] Implement memory retrieval functions
- [ ] Create navigation goal generation
- [ ] Develop temporal reasoning capabilities

### 5.3. Response Generation

- [ ] Set up response parsing and formatting
- [ ] Implement position and orientation extraction
- [ ] Create visualization tools for responses

## 6. Testing and Validation (Week 6)

### 6.1. Test Scenarios

- [ ] Create test suite with various query types:
  - Spatial queries: "Where did I see the red box?"
  - Temporal queries: "When did I last see a person?"
  - Action queries: "Take me to the loading dock."
  - Multi-step reasoning: "How long was I in the storage area?"

- [ ] Implement automated testing framework

### 6.2. Performance Benchmarking

- [ ] Measure memory retrieval latency
- [ ] Evaluate query response accuracy
- [ ] Benchmark system resource usage
- [ ] Test with increasing memory database size

### 6.3. Sim-to-Real Transfer Preparation

- [ ] Document optimal parameters
- [ ] Identify potential sim-to-real gap issues
- [ ] Create parameter export system for real robot

## 7. Integration with ROS 2 Navigation (Week 7)

### 7.1. Navigation Goal Integration

- [ ] Create ROS 2 action server for navigation goals
  ```python
  # Pseudocode for navigation integration
  def handle_navigation_query(query_text):
      # Get response from ReMEmbR agent
      response = agent.query(query_text)
      
      # Extract position for navigation
      if response.position is not None:
          # Send goal to navigation system
          send_navigation_goal(response.position)
          return True
      return False
  ```
- [ ] Implement callback system for navigation status

### 7.2. User Interface Development

- [ ] Create simple web interface using Gradio
  ```python
  import gradio as gr
  
  def process_query(query):
      response = agent.query(query)
      return response.text, visualize_position(response.position)
  
  interface = gr.Interface(
      fn=process_query,
      inputs="text",
      outputs=["text", "image"]
  )
  interface.launch()
  ```
- [ ] Add visualization components
- [ ] Implement real-time status updates

### 7.3. End-to-End System Testing

- [ ] Test complete workflow:
  1. User query input
  2. Agent processing
  3. Memory retrieval
  4. Response generation
  5. Navigation goal execution
- [ ] Verify system reliability over extended operation

## 8. Advanced Features and Optimizations (Week 8)

### 8.1. Memory Pruning and Management

- [ ] Implement memory pruning strategies
- [ ] Create hierarchical memory organization
- [ ] Optimize vector database queries

### 8.2. Simulation Randomization

- [ ] Implement domain randomization for sim-to-real transfer
  - Vary lighting conditions
  - Randomize object positions
  - Modify textures and colors
- [ ] Test system robustness under various conditions

### 8.3. Performance Optimization

- [ ] Profile system performance
- [ ] Optimize memory usage
- [ ] Reduce latency in critical paths
- [ ] Implement model quantization if needed

## 9. Documentation and Export (Week 9)

### 9.1. System Documentation

- [ ] Create detailed documentation of simulation setup
- [ ] Document API interfaces
- [ ] Create user manual for simulation environment

### 9.2. Parameter Export

- [ ] Create parameter export system for real robot
- [ ] Document optimal parameters
- [ ] Create transition guide for physical deployment

### 9.3. Final Validation

- [ ] Conduct comprehensive system validation
- [ ] Create demonstration scenarios
- [ ] Prepare for physical hardware transition

## Appendix: Required Software and Dependencies

### A.1. Primary Software

- NVIDIA Isaac Sim (latest version)
- ROS 2 Foxy or later
- NVIDIA CUDA Toolkit 12.x
- Docker for MilvusDB
- Conda for environment management

### A.2. Python Dependencies

```
# Core dependencies
numpy>=1.23.5
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.2
pillow>=9.5.0
opencv-python>=4.7.0.72
pymilvus>=2.2.8
langchain>=0.0.267
gradio>=3.35.2
rclpy  # For ROS 2 integration

# Isaac Sim specific
omni.isaac.core
omni.isaac.sensor
omni.isaac.ros2_bridge
```

### A.3. ReMEmbR Dependencies

- VILA (Vision-Language Assistant)
- Ollama for LLM support
- MilvusDB for vector database
- Command-R or other compatible LLM

### A.4. Hardware Requirements

- NVIDIA 4xxxx series GPU (12GB VRAM)
- 32GB+ RAM
- 500GB+ SSD storage 