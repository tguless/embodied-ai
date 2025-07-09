# ReMEmbR Implementation Guide

This document provides practical guidance for implementing and deploying a ReMEmbR system for robot navigation. It covers hardware and software requirements, installation steps, configuration options, and best practices for optimal performance.

## System Requirements

### Hardware Requirements

#### Minimum Requirements
- **Compute Platform**: NVIDIA Jetson Orin Nano (8GB) or equivalent
- **Memory**: 8GB RAM
- **Storage**: 32GB (SSD preferred)
- **Sensors**:
  - RGB camera (minimum 720p resolution)
  - Odometry source (wheel encoders, visual odometry, etc.)

#### Recommended Requirements
- **Compute Platform**: NVIDIA Jetson Orin 32GB or equivalent desktop/server with NVIDIA GPU
- **Memory**: 16GB+ RAM
- **Storage**: 128GB+ SSD
- **Sensors**:
  - RGB camera (1080p or higher resolution)
  - 3D LiDAR for accurate localization
  - Wheel encoders for odometry
  - IMU for motion tracking

### Software Requirements

#### Core Dependencies
- **Operating System**: Ubuntu 20.04 or newer
- **Robot Framework**: ROS2 Foxy or newer
- **Python**: 3.8 or newer
- **CUDA**: 11.4 or newer (for GPU acceleration)

#### Key Libraries
- **Vector Database**: FAISS or Chroma
- **Video Captioning**: VILA model (various sizes available)
- **Text Embedding**: mxbai-embed-large-v1 or alternatives
- **LLM Backend**: Options include:
  - Cloud-based: GPT-4o, NVIDIA NIM APIs
  - Local large models: Command-R, Codestral
  - On-device: Llama3.1-8b or similar function-calling LLMs

## Installation Guide

### 1. Environment Setup

Create a conda environment or virtual environment:

```bash
# Using conda
conda create -n remembr python=3.8
conda activate remembr

# Using venv
python3 -m venv remembr_env
source remembr_env/bin/activate
```

### 2. Install Core Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

# Install vector database
pip install faiss-gpu

# Install embedding model
pip install sentence-transformers

# Install ROS2 dependencies (if using ROS2)
# Follow ROS2 installation guide: https://docs.ros.org/en/foxy/Installation.html
```

### 3. Install VILA Model

```bash
# Install Hugging Face Transformers
pip install transformers

# Download VILA model (choose appropriate size)
# Options: VILA1.5-13b, VILA1.5-8b, VILA1.5-3b
python -c "from transformers import AutoProcessor, AutoModel; model = AutoModel.from_pretrained('NVIDIA/VILA-1.5-13b-hf'); processor = AutoProcessor.from_pretrained('NVIDIA/VILA-1.5-13b-hf')"
```

### 4. Install LLM Backend

Choose one of the following options:

#### Option A: Cloud-based LLM

```bash
# Install OpenAI API (for GPT-4o)
pip install openai

# Or install NVIDIA NIM API
pip install nvidia-nim
```

#### Option B: Local Large LLM

```bash
# Install libraries for local LLM inference
pip install vllm

# Download Command-R or Codestral model
# Follow instructions at: https://huggingface.co/CohereForAI/c4ai-command-r-v01
```

#### Option C: On-device Small LLM

```bash
# Install libraries for efficient on-device inference
pip install llama-cpp-python

# Download quantized Llama3.1-8b model
# Follow instructions at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

### 5. Install ReMEmbR

```bash
# Clone the ReMEmbR repository
git clone https://github.com/nvidia-ai-iot/remembr.git
cd remembr

# Install requirements
pip install -r requirements.txt

# Install the package
pip install -e .
```

## System Configuration

### Vector Database Setup

Create a configuration file `vector_db_config.yaml`:

```yaml
vector_db:
  type: "faiss"  # Options: "faiss", "chroma"
  dimension: 1024  # Embedding dimension
  index_type: "Flat"  # For FAISS, options: "Flat", "IVF", "HNSW"
  metric: "cosine"  # Distance metric: "cosine", "l2", "ip"
  storage_path: "/path/to/vector/storage"
```

### LLM Configuration

Create a configuration file `llm_config.yaml`:

```yaml
llm:
  type: "openai"  # Options: "openai", "nvidia", "vllm", "llama.cpp"
  
  # For OpenAI
  openai:
    api_key: "your_api_key"
    model: "gpt-4o"
    
  # For NVIDIA NIM
  nvidia:
    api_key: "your_api_key"
    model: "mixtral-8x7b"
    
  # For local vLLM
  vllm:
    model_path: "/path/to/model"
    max_tokens: 2048
    
  # For llama.cpp
  llama:
    model_path: "/path/to/model.gguf"
    n_ctx: 2048
    n_gpu_layers: -1  # -1 means all layers
```

### Video Captioning Configuration

Create a configuration file `captioning_config.yaml`:

```yaml
captioning:
  model: "VILA1.5-13b"  # Options: "VILA1.5-13b", "VILA1.5-8b", "VILA1.5-3b"
  segment_length: 3  # Seconds per segment
  fps: 2  # Frames per second to process
  batch_size: 1
  device: "cuda:0"  # Or "cpu" for CPU-only inference
```

## Implementation Steps

### 1. Memory Building Phase

Create a script `build_memory.py`:

```python
import os
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
import faiss
from PIL import Image
import yaml
import json
import time

# Load configurations
with open('captioning_config.yaml', 'r') as f:
    captioning_config = yaml.safe_load(f)

with open('vector_db_config.yaml', 'r') as f:
    vector_db_config = yaml.safe_load(f)

# Initialize VILA model
processor = AutoProcessor.from_pretrained(f"NVIDIA/{captioning_config['model']}-hf")
model = AutoModel.from_pretrained(f"NVIDIA/{captioning_config['model']}-hf")
model.to(captioning_config['device'])

# Initialize embedding model
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
embedding_model.to(captioning_config['device'])

# Initialize vector database
dimension = vector_db_config['dimension']
index = faiss.IndexFlatIP(dimension)  # Inner product similarity
memory_data = []  # Store metadata

def process_video_segment(frames, position, timestamp):
    """Process a segment of video frames and add to memory."""
    # Process frames with VILA
    inputs = processor(images=frames, return_tensors="pt").to(captioning_config['device'])
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Create embedding
    embedding = embedding_model.encode([caption])[0]
    embedding = embedding / np.linalg.norm(embedding)  # Normalize
    
    # Add to index
    index.add(np.array([embedding], dtype=np.float32))
    
    # Store metadata
    memory_data.append({
        'caption': caption,
        'position': position,
        'timestamp': timestamp,
        'index': len(memory_data)
    })
    
    return caption

def save_memory():
    """Save the vector database and metadata."""
    os.makedirs(vector_db_config['storage_path'], exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(vector_db_config['storage_path'], 'memory.index'))
    
    # Save metadata
    with open(os.path.join(vector_db_config['storage_path'], 'memory_data.json'), 'w') as f:
        json.dump(memory_data, f)

# Example usage in a ROS2 node or standalone script
# This would be integrated with your robot's perception pipeline
```

### 2. Querying Phase

Create a script `query_memory.py`:

```python
import os
import torch
import numpy as np
import faiss
import json
import yaml
from sentence_transformers import SentenceTransformer
import openai  # Or your chosen LLM API

# Load configurations
with open('vector_db_config.yaml', 'r') as f:
    vector_db_config = yaml.safe_load(f)

with open('llm_config.yaml', 'r') as f:
    llm_config = yaml.safe_load(f)

# Load vector database
index = faiss.read_index(os.path.join(vector_db_config['storage_path'], 'memory.index'))
with open(os.path.join(vector_db_config['storage_path'], 'memory_data.json'), 'r') as f:
    memory_data = json.load(f)

# Initialize embedding model
embedding_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')

# Initialize LLM
if llm_config['type'] == 'openai':
    openai.api_key = llm_config['openai']['api_key']

def text_retrieval(query, top_k=5):
    """Retrieve memories based on text similarity."""
    query_embedding = embedding_model.encode([query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
    
    # Search the index
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    
    # Get the memories
    retrieved_memories = [memory_data[int(idx)] for idx in indices[0]]
    return retrieved_memories

def position_retrieval(position, top_k=5):
    """Retrieve memories based on position proximity."""
    # Calculate distances to all positions
    distances = []
    for memory in memory_data:
        mem_pos = memory['position']
        dist = np.sqrt((position[0] - mem_pos[0])**2 + 
                       (position[1] - mem_pos[1])**2 + 
                       (position[2] - mem_pos[2])**2)
        distances.append((dist, memory))
    
    # Sort and return top_k
    distances.sort(key=lambda x: x[0])
    return [item[1] for item in distances[:top_k]]

def time_retrieval(timestamp, top_k=5):
    """Retrieve memories based on temporal proximity."""
    # Calculate time differences
    time_diffs = []
    for memory in memory_data:
        mem_time = memory['timestamp']
        diff = abs(timestamp - mem_time)
        time_diffs.append((diff, memory))
    
    # Sort and return top_k
    time_diffs.sort(key=lambda x: x[0])
    return [item[1] for item in time_diffs[:top_k]]

def query_llm(question, context):
    """Query the LLM with the question and context."""
    if llm_config['type'] == 'openai':
        response = openai.ChatCompletion.create(
            model=llm_config['openai']['model'],
            messages=[
                {"role": "system", "content": "You are a helpful robot assistant that answers questions based on your memory."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            functions=[
                {
                    "name": "text_retrieval",
                    "description": "Retrieve memories based on text similarity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The text to search for"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "position_retrieval",
                    "description": "Retrieve memories based on position proximity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "position": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "The [x, y, z] position to search near"
                            }
                        },
                        "required": ["position"]
                    }
                },
                {
                    "name": "time_retrieval",
                    "description": "Retrieve memories based on temporal proximity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timestamp": {
                                "type": "number",
                                "description": "The timestamp to search near"
                            }
                        },
                        "required": ["timestamp"]
                    }
                }
            ],
            function_call="auto"
        )
        return response
    
    # Add implementations for other LLM backends here
    
def answer_question(question, max_iterations=3):
    """Answer a question using the memory system."""
    context = ""
    retrieved_memories = []
    
    for i in range(max_iterations):
        # Query the LLM
        response = query_llm(question, context)
        
        # Check if the LLM wants to call a function
        if response.choices[0].message.get("function_call"):
            function_name = response.choices[0].message["function_call"]["name"]
            function_args = json.loads(response.choices[0].message["function_call"]["arguments"])
            
            # Call the appropriate function
            if function_name == "text_retrieval":
                new_memories = text_retrieval(function_args["query"])
            elif function_name == "position_retrieval":
                new_memories = position_retrieval(function_args["position"])
            elif function_name == "time_retrieval":
                new_memories = time_retrieval(function_args["timestamp"])
            
            # Add new memories to context
            for memory in new_memories:
                if memory not in retrieved_memories:
                    retrieved_memories.append(memory)
                    context += f"Memory {len(retrieved_memories)}: Caption: {memory['caption']}, Position: {memory['position']}, Timestamp: {memory['timestamp']}\n"
        else:
            # LLM has enough context to answer
            break
    
    # Generate final answer
    final_response = query_llm(question, context)
    return final_response.choices[0].message["content"]

# Example usage
# answer = answer_question("Where did you see the coffee machine?")
# print(answer)
```

## Integration with ROS2

### 1. Create a ROS2 Package

```bash
# Create a ROS2 workspace if you don't have one
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Create a package for ReMEmbR
ros2 pkg create --build-type ament_python remembr_ros
```

### 2. Implement ROS2 Nodes

Create a memory building node in `~/ros2_ws/src/remembr_ros/remembr_ros/memory_builder_node.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading
import queue

# Import your memory building functions
from remembr.memory import process_video_segment, save_memory

class MemoryBuilderNode(Node):
    def __init__(self):
        super().__init__('memory_builder')
        
        # Parameters
        self.declare_parameter('segment_length', 3.0)
        self.segment_length = self.get_parameter('segment_length').value
        
        self.declare_parameter('fps', 2.0)
        self.fps = self.get_parameter('fps').value
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
            
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        # Initialize buffers
        self.frame_buffer = []
        self.current_position = [0.0, 0.0, 0.0]
        self.frame_queue = queue.Queue()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.get_logger().info('Memory Builder Node initialized')
    
    def image_callback(self, msg):
        """Process incoming camera images."""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Add to queue with timestamp
            self.frame_queue.put((cv_image, time.time(), self.current_position.copy()))
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def odom_callback(self, msg):
        """Update current position from odometry."""
        self.current_position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]
    
    def process_frames(self):
        """Process frames in a separate thread."""
        last_process_time = time.time()
        frames = []
        positions = []
        timestamps = []
        
        while rclpy.ok():
            try:
                # Get frame from queue if available
                if not self.frame_queue.empty():
                    frame, timestamp, position = self.frame_queue.get(timeout=0.1)
                    frames.append(frame)
                    positions.append(position)
                    timestamps.append(timestamp)
                
                # Process if we have enough frames or enough time has passed
                current_time = time.time()
                if (len(frames) >= self.segment_length * self.fps or 
                    (current_time - last_process_time >= self.segment_length and len(frames) > 0)):
                    
                    # Average position and timestamp
                    avg_position = np.mean(positions, axis=0).tolist()
                    avg_timestamp = np.mean(timestamps)
                    
                    # Process the segment
                    caption = process_video_segment(frames, avg_position, avg_timestamp)
                    self.get_logger().info(f'Processed segment: {caption}')
                    
                    # Reset buffers
                    frames = []
                    positions = []
                    timestamps = []
                    last_process_time = current_time
                
                # Sleep to reduce CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                self.get_logger().error(f'Error in processing thread: {str(e)}')
    
    def on_shutdown(self):
        """Save memory when shutting down."""
        self.get_logger().info('Saving memory database...')
        save_memory()
        self.get_logger().info('Memory saved successfully')

def main(args=None):
    rclpy.init(args=args)
    node = MemoryBuilderNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Create a query node in `~/ros2_ws/src/remembr_ros/remembr_ros/query_node.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json

# Import your querying functions
from remembr.query import answer_question

class QueryNode(Node):
    def __init__(self):
        super().__init__('query_node')
        
        # Create subscribers
        self.query_sub = self.create_subscription(
            String,
            '/remembr/query',
            self.query_callback,
            10)
        
        # Create publishers
        self.answer_pub = self.create_publisher(
            String,
            '/remembr/answer',
            10)
        
        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10)
        
        self.get_logger().info('Query Node initialized')
    
    def query_callback(self, msg):
        """Process incoming queries."""
        try:
            question = msg.data
            self.get_logger().info(f'Received query: {question}')
            
            # Get answer from memory system
            answer = answer_question(question)
            
            # Publish text answer
            answer_msg = String()
            answer_msg.data = answer
            self.answer_pub.publish(answer_msg)
            
            # Check if answer contains coordinates
            try:
                # Parse answer to see if it contains coordinates
                parsed = json.loads(answer)
                if 'position' in parsed:
                    # Create and publish goal pose
                    goal_msg = PoseStamped()
                    goal_msg.header.frame_id = 'map'
                    goal_msg.header.stamp = self.get_clock().now().to_msg()
                    goal_msg.pose.position.x = parsed['position'][0]
                    goal_msg.pose.position.y = parsed['position'][1]
                    goal_msg.pose.position.z = parsed['position'][2]
                    goal_msg.pose.orientation.w = 1.0
                    
                    self.goal_pub.publish(goal_msg)
                    self.get_logger().info(f'Published goal: {parsed["position"]}')
            except:
                # Answer is not in JSON format or doesn't contain position
                pass
                
        except Exception as e:
            self.get_logger().error(f'Error processing query: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = QueryNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3. Build and Run

```bash
# Build the package
cd ~/ros2_ws
colcon build --packages-select remembr_ros

# Source the workspace
source install/setup.bash

# Run the nodes
ros2 run remembr_ros memory_builder_node
ros2 run remembr_ros query_node

# In another terminal, send a query
ros2 topic pub -1 /remembr/query std_msgs/String "data: 'Where did you see the coffee machine?'"
```

## Performance Optimization

### Memory Building Optimization

1. **Batch Processing**: Process multiple video segments in parallel when possible
   ```python
   # Example batch processing
   batch_size = 4  # Adjust based on available memory
   batch_frames = [frames1, frames2, frames3, frames4]
   batch_inputs = processor(images=batch_frames, return_tensors="pt").to(device)
   batch_outputs = model.generate(**batch_inputs)
   batch_captions = processor.batch_decode(batch_outputs, skip_special_tokens=True)
   ```

2. **Model Quantization**: Use quantized models for faster inference
   ```bash
   # Example for VILA model quantization
   pip install optimum
   python -m optimum.exporters.onnx --model NVIDIA/VILA-1.5-3b-hf vila_model_quantized
   ```

3. **Selective Captioning**: Only process segments with significant changes
   ```python
   # Example: Skip processing if frames are too similar
   if frame_similarity(previous_frame, current_frame) > 0.95:
       # Skip processing this frame
       continue
   ```

### Querying Optimization

1. **Caching**: Cache common queries and their results
   ```python
   # Simple query cache
   query_cache = {}
   
   def cached_answer_question(question):
       if question in query_cache:
           return query_cache[question]
       
       answer = answer_question(question)
       query_cache[question] = answer
       return answer
   ```

2. **Pre-filtering**: Use metadata to filter memories before embedding comparison
   ```python
   # Example: Pre-filter by time range
   def time_range_retrieval(start_time, end_time):
       filtered_memories = [m for m in memory_data if start_time <= m['timestamp'] <= end_time]
       return filtered_memories
   ```

3. **Progressive Loading**: Start with a small number of memories and increase if needed
   ```python
   # Start with fewer memories and increase if confidence is low
   initial_k = 3
   memories = text_retrieval(query, top_k=initial_k)
   
   # If confidence is low, retrieve more
   if confidence_score < 0.7:
       memories.extend(text_retrieval(query, top_k=10)[initial_k:])
   ```

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory Errors**
   - Reduce batch size
   - Use smaller models (VILA1.5-3b instead of VILA1.5-13b)
   - Increase swap space
   - Reduce video resolution or frame rate

2. **Slow Inference**
   - Enable GPU acceleration
   - Use quantized models
   - Reduce processing frequency
   - Use more efficient vector database indices (e.g., FAISS IVF or HNSW)

3. **Inaccurate Captions**
   - Fine-tune VILA model on domain-specific data
   - Increase video resolution
   - Adjust lighting conditions
   - Implement multi-view captioning

4. **Poor Retrieval Results**
   - Adjust similarity thresholds
   - Try different embedding models
   - Implement hybrid retrieval (combine text, position, and time)
   - Increase the number of retrieved memories

5. **Integration Issues with Navigation**
   - Verify coordinate frame transformations
   - Check for timestamp synchronization
   - Ensure proper ROS topic connections
   - Validate localization accuracy

## Best Practices

1. **Memory Management**
   - Regularly consolidate similar memories
   - Implement importance-based retention
   - Use hierarchical storage for different time scales
   - Periodically backup the vector database

2. **Deployment Strategy**
   - Start with small-scale testing in controlled environments
   - Gradually increase deployment duration
   - Monitor system resource usage
   - Implement graceful degradation for resource constraints

3. **Query Design**
   - Provide clear instructions to users about query formulation
   - Implement query reformulation for ambiguous queries
   - Support multi-modal queries when possible
   - Design fallback strategies for unanswerable queries

4. **Evaluation and Monitoring**
   - Regularly evaluate system performance on standard queries
   - Monitor caption quality and retrieval accuracy
   - Track response times and resource usage
   - Collect user feedback for continuous improvement

## Links to Related Documentation

- [ReMEmbR Architecture](remembr_architecture.md) - Detailed technical analysis
- [Experimental Results](experimental_results.md) - Performance evaluation
- [Real-World Applications](real_world_applications.md) - Deployment case studies
- [System Diagrams](system_diagrams.md) - Visual representations of system components 