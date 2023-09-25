Data scientists develop the code for each step of the ML pipelines in the algorithms root folder. 

The steps can be grouped in preprocessing, training, batch inference, and postprocessing (evaluation). 

In each group, multiple steps can be defined in corresponding subfolders, which contain a folder for the unit tests (including optional inputs and outputs), the main functions, the readme, and a Docker file in case of a custom container need. 

In addition to main, multiple code files can be hosted in the same folder. 

Common helper libraries for all the steps can be hosted in a shared library folder. 

The data scientists are responsible for the development of the unit tests because they own the logic of the steps, and ML engineers are responsible for the error handling enhancement and test coverage recommendation. The CI/CD pipeline is responsible for running the tests, building the containers automatically (if necessary), and packaging the multiple source code files.