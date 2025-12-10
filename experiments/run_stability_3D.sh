module use /g/data/xp65/public/modules
module load conda/analysis3-25.09
export ncpus=100
export nk=50
time mpirun -n $n python3 stability_3D.py --ncfls 10 --nk $nk --ncpus $ncpus
python3 stability_3D.py --ncfls 10 --nk $nk --ncpus $ncpus --plot