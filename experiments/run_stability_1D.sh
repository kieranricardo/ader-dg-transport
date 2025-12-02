export n=4
time mpirun -n $n python3 stability_1D.py --ncfls 128 --nk 100 --ncpus $n
python3 stability_1D.py --ncfls 128 --nk 100 --ncpus $n --plot