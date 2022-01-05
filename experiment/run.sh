# Generate random programs and profile DEAP.
python ./tools/deap/profile.py

# Convert the random programs generated by DEAP 
# into forms useable by the other relevant GP tools.
python ./tools/gpsy/program_converter.py

# Profile TensorGP.
# python ./tools/tensorgp/profile.py

# Profile Operon.

# Compute and output some relevant statistics.
python ./tools/stats.py