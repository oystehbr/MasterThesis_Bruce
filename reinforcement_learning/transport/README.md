# The Reinforcement Learning Framework 🔎
__Author: Øystein Høistad Bruce__

## Most important files:
### Route_opt_modified.py 🛣️
This file contains the main code, which serves as a backbone of the framework by managing updates to the environment. 
The code can be run interactively or in the background, depending on your needs. 
This code initializes the framework. 

### maps.py 🗺️
This file contains the code for creating and initializing the maps, graphs, and environments used in __Route_opt_modified.py__
This file is where the maps/graphs/environments are created. 

### dumpers.py 🛻
This file contains the dumper class, which manages all interactions between the dumper and the environment. This ensures a well-structured and organized codebase. 

### loaders.py 🏗️
This file contains the loader class, which is responsible for initializing the loaders within the environment. 
The management of the loaders is primarly handled by the loading node, which represents the physical location of the loader. 

### nodes_edges.py 🕸️
This file contains five classes: _Node_, _Dumping (Node)_, _Loading (Node)_, _Parking (Node)_, _Edge_.
- __Node__: This class represents a node (in the graph) within the environment and contains functionality that applies to all nodes. Intersection nodes are created using this class. 
- __Dumping__: This class is a subclass of __Node__ and represents a dumping node. It has additional properties that store all relevant information related to dumping mass at the node.
- __Loading__: This class is a subclass of __Node__ and represents a loading node. It has additional properties that store all relevant information related to loading mass at the node. In addition, it has a reference to the specific loader at the node. 
- __Parking__: This class is a subclass of __Node__ and represents a parking node. It has the additional property of being able to have dumpers parked there. 
- __Edge__: This class represents a connection between two nodes and contains information about the cost of traveling between them. 

### agent_(coffee_break, node2, plan2): 🕵🏼
These files contain the code for the different agents within the environment. 
While the structure of the agent classes is similar, eaech agent has its own unique set of states and actions. 

### game_modified.py 🎯
The file contains functions and information that are useful for reporting and tracking progress during an episode. 

### helper.py 🆘
The file contains functions that are used throughout the project but do not naturally fit within any other specific file. 
