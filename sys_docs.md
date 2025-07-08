# Geospatial LLM System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Core Components](#core-components)
5. [Implementation Guide](#implementation-guide)
6. [Usage Examples](#usage-examples)
7. [Evaluation & Testing](#evaluation--testing)
8. [Deployment](#deployment)

## System Overview

The Geospatial LLM System is a Chain-of-Thought-based framework that automatically generates and executes geospatial workflows from natural language queries. The system combines the reasoning capabilities of Large Language Models with robust geoprocessing APIs to solve complex spatial analysis tasks.

### Key Features
- **Natural Language to Workflow**: Convert user queries into executable GIS workflows
- **Chain-of-Thought Reasoning**: Transparent step-by-step reasoning process
- **Multi-format Support**: Handle vector, raster, and tabular geospatial data
- **Interactive Visualization**: Web-based interface with real-time results
- **Extensible Architecture**: Easy to add new tools and operations

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   LLM Agent     │───▶│  Geo Tools      │
│  (Natural Lang) │    │  (LangChain)    │    │ (GeoPandas/     │
└─────────────────┘    └─────────────────┘    │  Rasterio)      │
                                │              └─────────────────┘
                                ▼                       │
┌─────────────────┐    ┌─────────────────┐              │
│   Streamlit UI  │◀───│   Workflow      │◀─────────────┘
│  (Visualization)│    │   Executor      │
└─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   RAG System    │
                       │  (Chroma DB)    │
                       └─────────────────┘
```

### Core Components
1. **LLM Agent** - Reasoning and workflow planning
2. **RAG System** - Knowledge retrieval for GIS operations
3. **Workflow Executor** - Orchestrates geospatial operations
4. **Tool Registry** - Collection of available GIS functions
5. **Data Manager** - Handles input/output data operations
6. **UI Interface** - Web-based interaction and visualization

## Installation & Setup

### Prerequisites
- Python 3.8+
- 8GB+ RAM (for local LLM)
- GDAL/OGR libraries
- Git

### Environment Setup

```bash
# Create virtual environment
python -m venv geospatial-llm-env
source geospatial-llm-env/bin/activate  # On Windows: geospatial-llm-env\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev

# Install Ollama for local LLM
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral:7b-instruct
```

### Configuration

```python
# config/settings.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    # LLM Configuration
    LLM_MODEL: str = "mistral:7b-instruct"
    LLM_TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 2048
    
    # RAG Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_STORE_PATH: str = "data/vector_store"
    
    # Data Configuration
    DATA_PATH: str = "data/geospatial"
    OUTPUT_PATH: str = "outputs"
    
    # UI Configuration
    STREAMLIT_PORT: int = 8501
    DEBUG: bool = False
    
    class Config:
        env_file = ".env"
```

## Core Components

### 1. LLM Agent System

```python
# src/agent/geo_agent.py
from langchain.agents import Tool, AgentExecutor, create_structured_chat_agent
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class GeoSpatialAgent:
    def __init__(self, config):
        self.llm = Ollama(model=config.LLM_MODEL)
        self.memory = ConversationBufferMemory()
        self.tools = self._initialize_tools()
        self.agent = self._create_agent()
    
    def _initialize_tools(self):
        return [
            Tool(name="spatial_join", func=self.spatial_join_tool),
            Tool(name="buffer_analysis", func=self.buffer_analysis_tool),
            Tool(name="raster_analysis", func=self.raster_analysis_tool),
            Tool(name="site_selection", func=self.site_selection_tool),
        ]
    
    def generate_workflow(self, query: str) -> dict:
        """Generate workflow from natural language query"""
        prompt = f"""
        Task: Convert this geospatial query into a step-by-step workflow:
        Query: {query}
        
        Think step by step:
        1. What is the main objective?
        2. What data sources are needed?
        3. What operations need to be performed?
        4. What is the expected output?
        
        Return a structured workflow in JSON format.
        """
        
        response = self.agent.invoke({"input": prompt})
        return self._parse_workflow(response["output"])
```

### 2. RAG System Implementation

```python
# src/rag/knowledge_base.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, WebBaseLoader

class GeoKnowledgeBase:
    def __init__(self, config):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def build_knowledge_base(self):
        """Build RAG knowledge base from GIS documentation"""
        # Load documentation
        docs = self._load_documentation()
        
        # Split documents
        split_docs = self.text_splitter.split_documents(docs)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory="data/vector_store"
        )
    
    def retrieve_relevant_info(self, query: str, k: int = 5):
        """Retrieve relevant information for query"""
        if not self.vector_store:
            self.build_knowledge_base()
        
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in relevant_docs]
```

### 3. Workflow Executor

```python
# src/executor/workflow_executor.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import geopandas as gpd
import rasterio
from pathlib import Path

class WorkflowStep(BaseModel):
    operation: str
    parameters: Dict[str, Any]
    input_files: List[str]
    output_file: str
    reasoning: str

class WorkflowExecutor:
    def __init__(self, config):
        self.config = config
        self.tools = self._register_tools()
        self.execution_log = []
    
    def execute_workflow(self, workflow: List[WorkflowStep]) -> Dict[str, Any]:
        """Execute complete workflow"""
        results = {}
        
        for step in workflow:
            try:
                # Log reasoning
                self.execution_log.append({
                    "step": step.operation,
                    "reasoning": step.reasoning,
                    "status": "executing"
                })
                
                # Execute step
                result = self._execute_step(step)
                results[step.output_file] = result
                
                # Update log
                self.execution_log[-1]["status"] = "completed"
                
            except Exception as e:
                self.execution_log[-1]["status"] = f"failed: {str(e)}"
                raise
        
        return results
    
    def _execute_step(self, step: WorkflowStep):
        """Execute individual workflow step"""
        tool_func = self.tools.get(step.operation)
        if not tool_func:
            raise ValueError(f"Unknown operation: {step.operation}")
        
        return tool_func(step.parameters, step.input_files, step.output_file)
```

### 4. Geospatial Tools Registry

```python
# src/tools/geo_tools.py
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, Polygon
from typing import List, Dict, Any

class GeoTools:
    """Collection of geospatial analysis tools"""
    
    @staticmethod
    def spatial_join(params: Dict, inputs: List[str], output: str):
        """Perform spatial join between two datasets"""
        left_gdf = gpd.read_file(inputs[0])
        right_gdf = gpd.read_file(inputs[1])
        
        # Ensure same CRS
        right_gdf = right_gdf.to_crs(left_gdf.crs)
        
        # Perform spatial join
        result = gpd.sjoin(left_gdf, right_gdf, 
                          how=params.get('how', 'inner'),
                          predicate=params.get('predicate', 'intersects'))
        
        result.to_file(output)
        return {"status": "success", "records": len(result)}
    
    @staticmethod
    def buffer_analysis(params: Dict, inputs: List[str], output: str):
        """Create buffer around geometries"""
        gdf = gpd.read_file(inputs[0])
        
        # Create buffer
        buffer_distance = params.get('distance', 1000)
        buffered = gdf.copy()
        buffered.geometry = gdf.geometry.buffer(buffer_distance)
        
        buffered.to_file(output)
        return {"status": "success", "buffer_distance": buffer_distance}
    
    @staticmethod
    def raster_clip(params: Dict, inputs: List[str], output: str):
        """Clip raster by polygon"""
        raster_path = inputs[0]
        polygon_path = inputs[1]
        
        # Read polygon
        polygon_gdf = gpd.read_file(polygon_path)
        
        # Clip raster
        with rasterio.open(raster_path) as src:
            # Reproject polygon to raster CRS
            polygon_reproj = polygon_gdf.to_crs(src.crs)
            
            # Clip
            clipped_data, clipped_transform = mask(
                src, polygon_reproj.geometry, crop=True
            )
            
            # Save result
            profile = src.profile
            profile.update({
                'height': clipped_data.shape[1],
                'width': clipped_data.shape[2],
                'transform': clipped_transform
            })
            
            with rasterio.open(output, 'w', **profile) as dst:
                dst.write(clipped_data)
        
        return {"status": "success", "clipped": True}
```

## Implementation Guide

### Phase 1: Core System Setup (Week 1-2)

1. **Environment Setup**
   ```bash
   # Set up project structure
   mkdir geospatial-llm-system
   cd geospatial-llm-system
   
   # Initialize git repository
   git init
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install langchain ollama geopandas rasterio streamlit
   pip install chromadb sentence-transformers pydantic
   ```

3. **Basic LLM Integration**
   ```python
   # Test Ollama connection
   from langchain.llms import Ollama
   
   llm = Ollama(model="mistral:7b-instruct")
   response = llm("What are the steps for flood risk analysis?")
   print(response)
   ```

### Phase 2: RAG System Development (Week 3-4)

1. **Documentation Collection**
   - Download QGIS documentation
   - Collect GDAL/OGR references
   - Create custom GIS operation guides

2. **Vector Store Creation**
   ```python
   # Build knowledge base
   kb = GeoKnowledgeBase(config)
   kb.build_knowledge_base()
   ```

### Phase 3: Workflow Generation (Week 5-6)

1. **Prompt Engineering**
   ```python
   WORKFLOW_PROMPT = """
   You are a GIS expert. Convert this query into a step-by-step workflow:
   
   Query: {query}
   
   Available tools:
   - spatial_join: Join datasets based on spatial relationship
   - buffer_analysis: Create buffer zones around features
   - raster_clip: Clip raster data using polygon boundaries
   
   Return JSON format with steps, reasoning, and parameters.
   """
   ```

2. **Chain-of-Thought Implementation**
   ```python
   def generate_workflow_with_reasoning(query):
       reasoning_prompt = f"""
       Task: {query}
       
       Let me think step by step:
       1. What is the main objective?
       2. What data do I need?
       3. What operations are required?
       4. What is the expected output?
       
       Based on this analysis, here's my workflow:
       """
       
       return llm.invoke(reasoning_prompt)
   ```

### Phase 4: Tool Integration (Week 7-8)

1. **Geospatial Operations**
   - Implement core GIS functions
   - Add error handling and validation
   - Create tool registration system

2. **Testing Framework**
   ```python
   # Test individual tools
   def test_spatial_join():
       result = GeoTools.spatial_join(
           params={'how': 'inner'},
           inputs=['data/points.shp', 'data/polygons.shp'],
           output='output/joined.shp'
       )
       assert result['status'] == 'success'
   ```

### Phase 5: UI Development (Week 9-10)

1. **Streamlit Interface**
   ```python
   import streamlit as st
   
   st.title("Geospatial LLM System")
   
   # Query input
   query = st.text_area("Enter your geospatial analysis query:")
   
   if st.button("Generate Workflow"):
       # Process query
       workflow = agent.generate_workflow(query)
       
       # Display reasoning
       st.subheader("Chain of Thought")
       st.write(workflow['reasoning'])
       
       # Execute workflow
       results = executor.execute_workflow(workflow['steps'])
       
       # Display results
       st.subheader("Results")
       for output_file, result in results.items():
           st.write(f"Generated: {output_file}")
   ```

## Usage Examples

### Example 1: Flood Risk Assessment

```python
query = """
I need to assess flood risk for residential areas near the Ganges River. 
I have elevation data, river shapefile, and residential zones. 
Create a flood risk map showing areas within 500m of the river 
that are below 10m elevation.
"""

# Expected workflow:
# 1. Buffer river by 500m
# 2. Clip elevation raster to buffer area
# 3. Identify areas below 10m elevation
# 4. Intersect with residential zones
# 5. Create risk classification map
```

### Example 2: Site Suitability Analysis

```python
query = """
Find suitable locations for solar farms. Requirements:
- Flat terrain (slope < 5 degrees)
- Away from residential areas (>1km)
- Good road access (<500m from roads)
- Minimum 10 hectares area
"""

# Expected workflow:
# 1. Calculate slope from DEM
# 2. Filter areas with slope < 5 degrees
# 3. Buffer residential areas by 1km (exclusion zone)
# 4. Buffer roads by 500m (inclusion zone)
# 5. Combine criteria and filter by minimum area
```

## Evaluation & Testing

### Automated Testing Framework

```python
# tests/test_workflows.py
import pytest
from src.agent.geo_agent import GeoSpatialAgent
from src.executor.workflow_executor import WorkflowExecutor

class TestWorkflows:
    def test_flood_risk_workflow(self):
        """Test flood risk assessment workflow"""
        query = "Assess flood risk for areas near river"
        
        # Generate workflow
        workflow = self.agent.generate_workflow(query)
        
        # Validate workflow structure
        assert 'steps' in workflow
        assert len(workflow['steps']) > 0
        
        # Execute workflow
        results = self.executor.execute_workflow(workflow['steps'])
        
        # Validate results
        assert all(result['status'] == 'success' for result in results.values())
    
    def test_chain_of_thought_clarity(self):
        """Test reasoning clarity and traceability"""
        query = "Find suitable locations for wind farms"
        
        workflow = self.agent.generate_workflow(query)
        
        # Check reasoning quality
        reasoning = workflow['reasoning']
        assert len(reasoning) > 100  # Minimum reasoning length
        assert 'step' in reasoning.lower()
        assert 'because' in reasoning.lower()
```

### Performance Benchmarks

```python
# benchmarks/performance_test.py
import time
import psutil
from src.benchmarks.baseline_comparison import BaselineComparison

def benchmark_workflow_generation():
    """Benchmark workflow generation performance"""
    test_queries = [
        "Assess flood risk near river",
        "Find suitable agricultural land",
        "Analyze urban growth patterns"
    ]
    
    results = {}
    
    for query in test_queries:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # Generate workflow
        workflow = agent.generate_workflow(query)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        results[query] = {
            'generation_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'workflow_steps': len(workflow['steps'])
        }
    
    return results
```

## Deployment

### Local Development
```bash
# Start the application
streamlit run src/ui/app.py --server.port 8501
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Production Considerations

1. **Scalability**
   - Use Redis for caching
   - Implement job queuing for long-running workflows
   - Load balancing for multiple instances

2. **Security**
   - Input validation and sanitization
   - Rate limiting for API endpoints
   - Secure file upload handling

3. **Monitoring**
   - Logging for all operations
   - Performance metrics collection
   - Error tracking and alerting

## Troubleshooting

### Common Issues

1. **Memory Issues with LLM**
   ```python
   # Reduce model size or use quantized models
   llm = Ollama(model="mistral:7b-instruct-q4_0")
   ```

2. **CRS Mismatches**
   ```python
   # Always check and align CRS
   if gdf1.crs != gdf2.crs:
       gdf2 = gdf2.to_crs(gdf1.crs)
   ```

3. **Large File Handling**
   ```python
   # Use chunked processing for large datasets
   def process_large_raster(raster_path, chunk_size=1024):
       with rasterio.open(raster_path) as src:
           for window in src.block_windows(1):
               chunk = src.read(window=window)
               # Process chunk
   ```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-tool`
3. Commit changes: `git commit -am 'Add new geospatial tool'`
4. Push to branch: `git push origin feature/new-tool`
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.