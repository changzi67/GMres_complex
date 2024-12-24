cmake --build build/ --parallel
# build/ilu_gmres -m data/test/test.mtx -d 1e-6 -e 500 -p 1 -c 1 -i 1000
# build/ilu_gmres -m data/test/test.mtx -c 2 -n 4

# build/ilu_gmres -m data/Chevron4/Chevron4.mtx -c 1 -d 1e-6 -e 50 -p 0.2 # Success
# build/ilu_gmres -m data/Chevron4/Chevron4.mtx -c 2 -n 5 -r 200 # Failed, res ~ 2e-4

# build/ilu_gmres -m data/dielFilterV2clx/dielFilterV2clx.mtx -b data/dielFilterV2clx/dielFilterV2clx_b.mtx -c 1 -d 1e-4 -e 50 -p 0.2 # Failed
# build/ilu_gmres -m data/dielFilterV2clx/dielFilterV2clx.mtx -b data/dielFilterV2clx/dielFilterV2clx_b.mtx -c 2 -n 5 -r 200 # res ~ 0.2 

# build/ilu_gmres -m data/fem_hifreq_circuit/fem_hifreq_circuit.mtx -c 1 -d 3e-4 -e 50 -p 0.01  # Very Slow and not converge
# build/ilu_gmres -m data/fem_hifreq_circuit/fem_hifreq_circuit.mtx -c 2 -n 5 -r 400 -i 25 # when r gets larger, it gets Slower, relres ~ 2.5e-6

# build/ilu_gmres -m data/mono_500Hz/mono_500Hz.mtx -c 1 -d 1e-4 -e 80 -p 0.01  # slow and not converge
# build/ilu_gmres -m data/mono_500Hz/mono_500Hz.mtx -c 2 -n 5 # much faster and effective res ~ 1e-5