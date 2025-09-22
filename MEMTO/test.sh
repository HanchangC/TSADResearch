# SMD
python main.py --anormly_ratio 0.5 --num_epochs 50 --batch_size 256 --mode train --dataset SMD --data_path ./data/SMD --input_c 38 --output_c 38 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

python main.py --anormly_ratio 0.5 --num_epochs 50  --batch_size 256  --mode memory_initial --dataset SMD  --data_path ./data/SMD  --input_c 38 --output_c 38 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# test
python main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 256  --mode test --dataset SMD  --data_path ./data/SMD --input_c 38 --output_c 38 --n_memory 10 --memory_initial False --phase_type test


# # # SMAP
python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 256 --mode train --dataset SMAP --data_path ./data/SMAP --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

python main.py --anormly_ratio 1.0 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset SMAP  --data_path ./data/SMAP  --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# test
python main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 256  --mode test --dataset SMAP  --data_path ./data/SMAP  --input_c 25 --output_c 25 --n_memory 10 --memory_initial False --phase_type test

# # PSM
python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 256 --mode train --dataset PSM --data_path ./data/PSM --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

python main.py --anormly_ratio 1.0 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset PSM  --data_path ./data/PSM  --input_c 25 --output_c 25 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# test
python main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 256  --mode test --dataset PSM  --data_path ./data/PSM --input_c 25 --output_c 25 --n_memory 10 --memory_initial False --phase_type test

# MSL

python main.py --anormly_ratio 1.0 --num_epochs 100 --batch_size 256 --mode train --dataset MSL --data_path ./data/MSL --input_c 55 --output_c 55 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

python main.py --anormly_ratio 1.0 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset MSL  --data_path ./data/MSL  --input_c 55 --output_c 55 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# test
python main.py --anormly_ratio 1.0 --num_epochs 10   --batch_size 256  --mode test --dataset MSL  --data_path ./data/MSL --input_c 55 --output_c 55 --n_memory 10 --memory_initial False --phase_type test


# # SWaT

# python main.py --anormly_ratio 0.1 --num_epochs 100 --batch_size 256 --mode train --dataset SWaT --data_path ./data/SWaT/SWaT/ --input_c 51 --output_c 51 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anormly_ratio 0.1 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset SWaT  --data_path ./data/SWaT/SWaT/  --input_c 51 --output_c 51 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anormly_ratio 0.1 --num_epochs 100   --batch_size 256  --mode test --dataset SWaT  --data_path ./data/SWaT/SWaT/  --input_c 51 --output_c 51 --n_memory 10 --memory_initial False --phase_type test

# # NAB

# python main.py --anormly_ratio 0.1 --num_epochs 100 --batch_size 256 --mode train --dataset NAB --data_path ./data/NAB --input_c 1 --output_c 1 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anormly_ratio 0.1 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset NAB  --data_path ./data/NAB  --input_c 1 --output_c 1 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anormly_ratio 0.1 --num_epochs 100   --batch_size 256  --mode test --dataset NAB  --data_path ./data/NAB  --input_c 1 --output_c 1 --n_memory 10 --memory_initial False --phase_type test

# Yahoo

# python main.py --anormly_ratio 0.1 --num_epochs 100 --batch_size 256 --mode train --dataset Yahoo --data_path ./data/Yahoo --input_c 1 --output_c 1 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anormly_ratio 0.1 --num_epochs 100  --batch_size 256  --mode memory_initial --dataset Yahoo  --data_path ./data/Yahoo  --input_c 1 --output_c 1 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anormly_ratio 0.1 --num_epochs 100   --batch_size 256  --mode test --dataset Yahoo  --data_path ./data/Yahoo  --input_c 1 --output_c 1 --n_memory 10 --memory_initial False --phase_type test

# # AIOPS

# python main.py --anormly_ratio 0.1 --num_epochs 10 --batch_size 16 --mode train --dataset AIOPS --data_path ./data/AIOPS --input_c 1 --output_c 1 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anormly_ratio 0.1 --num_epochs 10  --batch_size 16  --mode memory_initial --dataset AIOPS  --data_path ./data/AIOPS  --input_c 1 --output_c 1 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anormly_ratio 0.1 --num_epochs 10   --batch_size 16  --mode test --dataset AIOPS  --data_path ./data/AIOPS  --input_c 1 --output_c 1 --n_memory 10 --memory_initial False --phase_type test

# # WSD

# python main.py --anormly_ratio 0.1 --num_epochs 5 --batch_size 16 --mode train --dataset WSD --data_path ./data/WSD --input_c 1 --output_c 1 --n_memory 10 --lambd 0.01 --lr 1e-4 --memory_initial False --phase_type None 

# python main.py --anormly_ratio 0.1 --num_epochs 5  --batch_size 16  --mode memory_initial --dataset WSD  --data_path ./data/WSD  --input_c 1 --output_c 1 --n_memory 10 --lambd 0.01 --lr 5e-5 --memory_initial True --phase_type second_train

# # test
# python main.py --anormly_ratio 0.1 --num_epochs 5   --batch_size 16  --mode test --dataset WSD  --data_path ./data/WSD  --input_c 1 --output_c 1 --n_memory 10 --memory_initial False --phase_type test