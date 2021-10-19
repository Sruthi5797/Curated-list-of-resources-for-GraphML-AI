# Curated-list-of-resources-for-GraphML-AI
![image](https://user-images.githubusercontent.com/42236363/134605006-f772e717-6e72-4ef7-98ce-cac8b01f342f.png)


Machine learning(ML) a sub-group of Artifical Intelligence(AI) is widely used in solving a wide range of real-world problems. They use various forms of data to address them of which graph data is one of them. Neo4j is one of the database used to store this graph data, also it is easy to analyse by establising relationship(edges) between the data often denoted as nodes.


In this list of we propose resources which are helpful for ML/AI on graphs that contains categories which are provided as below
| Sno | Title | Description |
| --- | -------|------------- |
| [1](#research-articles) | Research Articles | Research publications in Journal, Conferences etc.,|
| [2](#software-libraries) | Software Libraries | Libraries that integrate ML/AI with graphs |
| [3](#graph-packages) | Graph packages | Pacakges used to create and visualize graphs |
| [4](#books) | Books | Books on AI/ML for graphs |
| [5](#study-materials) | Workshops/Tutorials/Courses | Learning material for graphs |
| [6](#others) | Others | Code bases, Datasets, Projects and few blog posts on GraphML |


## Research articles
1. **Survey**
- [A Comprehensive Survey on Graph Neural Networks (2019)](https://arxiv.org/pdf/1901.00596.pdf)
- [A Practical Guide to Graph Neural Networks (2020)](https://arxiv.org/pdf/2010.05234.pdf)
- [Graph Neural Networks for Natural Language Processing: A Survey](https://arxiv.org/abs/2106.06090.pdf)
- [A systematic literature review of graph-based anomaly detection approaches (2020)](https://www.sciencedirect.com/science/article/pii/S0167923620300580)

2. **Data Preprocessing**
- Data Preprocessing [Local Augmentation for Graph Neural Networks](https://arxiv.org/abs/2109.03856)
- Data Augmentation [FLAG: Adversarial Data Augmentation for Graph Neural Networks](https://arxiv.org/pdf/2010.09891.pdf)

3. **Architecture**
- Knowledge graphs: [A Comprehensive Introduction to Knowledge Graphs (2021)](https://arxiv.org/pdf/2003.02320.pdf)
- K-associated graphs: [A nonparametric classification method based on K-associated graphs](https://sites.icmc.usp.br/alneu/papers/infoSciences2011.pdf)
- GCN for different datasets: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907.pdf)
- GATs for different datasets: [Graph Attention Networks](https://arxiv.org/abs/1710.10903.pdf)
- Graph LSTM [ Semantic Object Parsing with Graph LSTM](https://arxiv.org/pdf/1603.07063.pdf)
- Graph ST-GCN [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://github.com/yysijie/st-gcn)
- MPNN - can be used for other novel approaches [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)
- Learning CNN [Learning Convolutional Neural Networks for Graphs](https://arxiv.org/pdf/1605.05273.pdf)
- GG-NN [GATED GRAPH SEQUENCE NEURAL NETWORKS](https://arxiv.org/pdf/1511.05493.pdf)
- Custom network [DEEP GRAPH INFOMAX](https://arxiv.org/abs/1809.10341)
- Representation learning [Representation Learning on Graphs: Methods and Applications](https://arxiv.org/pdf/1709.05584.pdf)
- GDL review on sparse structures [Geometric deep learning: going beyond Euclidean data](https://arxiv.org/abs/1611.08097)
- Semi-supervised [Relating Graph Neural Networks to Structural Causal Models](https://arxiv.org/pdf/2109.04173.pdf)
- Graph explainability [Reimagining GNN Explanations with ideas from Tabular Data](https://arxiv.org/pdf/2106.12665.pdf)
- NLS approach [Learning to Generate Scene Graph from Natural Language Supervision](https://arxiv.org/pdf/2109.02227.pdf)
- MGNN for text summarization [Multiplex Graph Neural Network for Extractive Text Summarization](https://arxiv.org/pdf/2108.12870.pdf)
- TabGNN for prediction [TabGNN: Multiplex Graph Neural Network for Tabular Data Prediction](https://arxiv.org/pdf/2108.09127.pdf)
- AdaGNN for 4 datasets [AdaGNN: A multi-modal latent representation meta-learner for GNNs based on AdaBoosting](https://arxiv.org/pdf/2108.06452.pdf)
- X-GGM for VQA tasks [X-GGM: Graph Generative Modeling for Out-of-Distribution Generalization in Visual Question Answering](https://arxiv.org/abs/2107.11576)

4. **Application - link prediction**
- Link prediction using VGAE: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308.pdf)
- Link prediction and Entity Classification using R-GCN [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/pdf/1703.06103.pdf)
- Link prediction using node2vec [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653.pdf)
- Link prediction using DECAGON [Modeling polypharmacy side effects with graph convolutional networks](https://arxiv.org/pdf/1802.00543.pdf)

5. **Application - Others**
- Context Aware Security monitoring: [Machine learning on knowledge graphs for context-aware security monitoring](https://arxiv.org/pdf/2105.08741.pdf)
- Recommandation system: [Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/abs/1806.01973.pdf)
- Fake detection [Fake News Detection on Social Media using Geometric Deep Learning](https://arxiv.org/abs/1902.06673)
- Molecular finger printing [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292)
- Life-science application [DGL-LIFESCI: AN OPEN-SOURCE TOOLKIT FOR DEEP LEARNING ON GRAPHS IN LIFE SCIENCE](https://arxiv.org/pdf/2106.14232.pdf)
- Graph based forecasting [A Study of Joint Graph Inference and Forecasting](https://arxiv.org/abs/2109.04979)
- Drug discovery [HyperFoods: Machine intelligent mapping of cancer-beating molecules in foods](https://www.nature.com/articles/s41598-019-45349-y)

6. **Papers Relevant to systems engineering**
- Cyber physical systems: [Graph-Based Digital Blueprint for Model Based Engineering of Complex Systems](https://www.omgsysml.org/Graphs_MBE_INCOSE_IS_Bajaj-et-al.pdf)
- Model Based System Engineering: [Addressing Model-Based System Engineering Challenges with Knowledge Graphs](https://www.stardog.com/resources/addressing-model-based-system-engineering-challenges-with-knowledge-graphs/)
- 

6. **Dynamic Graphs**
- [Software Engineering Event Modeling using Relative Time in Temporal Knowledge Graphs](https://arxiv.org/pdf/2007.01231.pdf)
- [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/pdf/2006.10637.pdf)
- [Inductive Representation Learning on Temporal Graphs](https://arxiv.org/pdf/2002.07962.pdf)

## Software libraries
- Integrating ML with knowledge graphs: [Grakn KGLIB](https://github.com/vaticle/kglib)
- An ultimate library for GNN: [PvG](https://www.pyg.org/)
- Machine learning on graphs and networks: [StellarGraph](https://github.com/stellargraph/stellargraph)
- A library to perform deep learning on graphs. Contains a wide range of tutorial to start from:[Deep-Graph-Library](https://www.dgl.ai/)
- A framework for industive representation learning for large graphs: [GraphSAGE](http://snap.stanford.edu/graphsage/) <sup>[2](#myfootnote2)</sup>
- A simplified version of tensorflow for performing computation on graphs in a more transparent way [GraphAttack](https://github.com/jgolebiowski/graphAttack)
- An AI-toolkit for graph analysis, ML and NLP [Grasp](https://github.com/textgain/grasp)

## Graph packages
- Python package for creating, manipulating, and studying complex networks' structure, dynamics, and functions: [NetworkX](https://networkx.github.io/)
- Collection of network analysis with a focus on easy usabilitz, efficiency and portability: [igraph](https://igraph.org/)
- Graph manipulation,speed of processing, supports visualization and based on Boost graph library: [graph-tool](https://graph-tool.skewed.de/)
- Analysis and manipulation of networks by Stanford: [SNAP](https://snap.stanford.edu/snap/)
- Graph analysis using Julia: [LightGraphs](https://juliagraphs.org/LightGraphs.jl/latest/)
- A graph database for storing and retreiving data used in computational toxicology [ComptoxAI](https://comptox.ai/) <sup>[1](#myfootnote1)</sup>

## Books
- Graph Machine Learning: Take Graph Data to the Next Level by Applying Machine Learning Techniques and Algorithms
- Designing and Building Enterprise Knowledge Graphs (Synthesis Lectures on Data, Semantics, and Knowledge)
- Connecting the Dots:Harness the Power of Graphs & ML
- Ma, Y., & Tang, J. (2021). Deep Learning on Graphs. Cambridge: Cambridge University Press. doi:10.1017/9781108924184

## Study materials
- A Workshop on Graph Powered ML with presentation and tutorials: [Graph-Powered-ML](https://github.com/joerg84/Graph_Powered_ML_Workshop)
- Practical courses: [Learning-Networks-with-ML](https://github.com/Networks-Learning/mlss-2016)
- A Workshop on graph learning [Snap Workshop](https://snap.stanford.edu/graphlearning-workshop/index.html)

## Others
- List of GNNs available: [Awesome efficient GNN](https://github.com/chaitjo/awesome-efficient-gnn)
- A very good blog to understand GraphDL [GDL](https://ericmjl.github.io/essays-on-data-science/machine-learning/graph-nets/)
- List of resources for Graph bases reasoning [Graph Reasoning](https://github.com/AstraZeneca/awesome-explainable-graph-reasoning)
- List of Graph deep learning based papers [ICML 2021](https://github.com/naganandy/graph-based-deep-learning-literature/blob/master/conference-publications/folders/publications_icml21/README.md)
- Code examples of GraphNLP tutorials [graph-for-NLP](https://github.com/graph4ai/graph4nlp_demo)
- A library for open graph benchmarking [OGB](https://ogb.stanford.edu/)
- A GNN challende [AI/ML for GNN](https://arxiv.org/pdf/2107.12433.pdf)

### Data generation
- Library: [Synthetic-Graph-Data-Generation-for-ML](https://github.com/Octavian-ai/synthetic-graph-data)

### Graph datasets 
- A list of open-source graph datasets [Graph-datasets](https://github.com/AntonsRuberts/graph_ml)
- A list of datasets in GraphML[GraphML-datasets](https://github.com/yuehhua/GraphMLDatasets.jl)
- Graph reasoning dataset [CLEVR-graph](https://github.com/davidsketchdeck/clevr-graph)
- HPC analytics project data uses Neo4j and Python to run various ML algorithms [Property database](https://github.com/happystep/HPC_Analytics)
- Curated list of databases [Github-link](https://github.com/jbmusso/awesome-graph)
### Projects
- Fraud Detection in BankSim dataset containing bank transactions [Fraud-Detection](https://github.com/aravind-sundaresan/Graph-ML-Fraud-Detection)
### Available graphDBs <sup>[src](#myfootnote3)</sup>
* [AgensGraph](https://bitnine.net/agensgraph-2/) - multi-model graph database with SQL and Cypher support
* [AnzoGraph](https://www.cambridgesemantics.com/anzograph/) - Massively parallel graph database with advanced analytics (SPARQL, Cypher, OWL/RDFS+, LPG) 
* [Atomic-Server](https://crates.io/crates/atomic-server/) - open-source type-safe graph database server with GUI, written in rust. Supports [Atomic Data](docs.atomicdata.dev/), JSON & RDF.
* [ArangoDB](https://www.arangodb.com/) - highly available Multi-Model NoSQL database
* [Blazegraph](https://github.com/blazegraph/database) - GPU accelerated graph database
* [Cayley](https://github.com/cayleygraph/cayley) - open source database written in Go
* [CosmosDB](https://docs.microsoft.com/en-us/azure/cosmos-db/graph-introduction) - cloud-based multi-model database with support for TinkerPop3
* [Dgraph](https://dgraph.io) - Fast, Transactional, Distributed Graph Database (open source, written in Go)
* [DSE Graph](https://www.datastax.com/products/datastax-enterprise-graph) - Graph layer on top of DataStax Enterprise (Cassandra, SolR, Spark)
* [Grafito](https://github.com/arturo-lang/grafito) - Portable, Serverless & Lightweight SQLite-based Graph Database in Arturo
* [Grakn.AI](https://grakn.ai/) - a distributed hyper-relational database for knowledge-oriented systems, i.e. a distributed knowledge base
* [Graphd](https://github.com/google/graphd) - the Metaweb/Freebase Graph Repository
* [JanusGraph](http://janusgraph.org) - an open-source, distributed graph database with pluggable storage and indexing backends
* [Memgraph](https://memgraph.com/) - High Performance, In-Memory, Transactional Graph Database
* [Neo4j](http://tinkerpop.apache.org/docs/current/#neo4j-gremlin) - OLTP graph database
* [Nebula Graph](http://nebula-graph.io/) - A distributed, fast open-source graph database featuring horizontal scalability and high availability
* [RedisGraph](https://oss.redislabs.com/redisgraph/) - Property graph database, based on linear algebra constructs (GraphBLAS)
* [Sparksee](http://www.sparsity-technologies.com/#sparksee) - makes space and performance compatible with a small footprint and a fast analysis of large networks
* [Stardog](http://stardog.com/) - RDF graph database with OLTP and OLAP support
* [OrientDB](http://orientdb.com/orientdb/) - Distributed Multi-Model NoSQL Database with a Graph Database Engine
* [TerminusDB](https://github.com/terminusdb/terminusdb) is an open source graph database and document store. It is designed for collaboratively building data-intensive applications and knowledge graphs.
* [TigerGraph](https://www.tigergraph.com/) - The First Native Parallel Graph capable of real-time analytics on web-scale data
* [Weaviate](https://github.com/semi-technologies/weaviate) - Weaviate is a cloud-native, modular, real-time vector search engine with a graph data model (GraphQL interface) built to scale your machine learning models.

<a name="myfootnote1">1</a>: Support for ML coming soon <br>
<a name="myfootnote3">src</a>: https://raw.githubusercontent.com/jbmusso/awesome-graph/<br>
<a name="myfootnote2">2</a>: Also available as a beta version in Neo4J for node embeddings: [Neo4jGraphSAGE](https://neo4j.com/docs/graph-data-science/current/algorithms/graph-sage/)
