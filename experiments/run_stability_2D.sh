module use /g/data/hh5/public/modules
module load conda/analysis3-24.04
export n=100
export nk=100
time mpirun -n $n python3 stability_2D.py --nk $nk --ncpus $n
python3 stability_2D.py --nk $nk --ncpus $n --plot