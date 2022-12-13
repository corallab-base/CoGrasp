# CoGrasp

## CoGrasp - Main file is grasping.py, which uses [PoinTr](https://github.com/yuxumin/PoinTr) to complete the PointCloud for the observed objects in the scene and [ContactGraspnet](https://github.com/NVlabs/contact_graspnet) to generate intial set of grasps for the set of objects in picture. 
Inspired from [GraspTTA](https://github.com/hwjiang1510/GraspTTA), hand orientation is generated which represents the standard/reliable way in which human grabs the interested oject. PruningNetwork is used to select aprropriate grasps that are human-aware.

Please refer to [PoinTr](https://github.com/yuxumin/PoinTr), [ContactGraspnet](https://github.com/NVlabs/contact_graspnet) and [GraspTTA](https://github.com/hwjiang1510/GraspTTA) for setting it up.
