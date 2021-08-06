Installation
=========

Assuming you have access to the QuantCo DIL conda repository, you can install the package through conda
::

   # Set up the quantco_main conda channel. For the password, substitute in the correct password. You should be able to get the password by searching around on slack or asking on the glm_benchmarks slack channel!
   conda config --system --prepend channels quantco_main
   conda config --system --set custom_channels.quantco_main https://dil_ro:password@conda.quantco.cloud

   conda install quantcore.glm

