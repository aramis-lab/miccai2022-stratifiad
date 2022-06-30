echo "Starting training for test 03 in cuda:1"

python src/train.py -c 'configs/experiment_002_128x128/config_test_03_cv_00.yaml'
sleep 30

python src/train.py -c 'configs/experiment_002_128x128/config_test_03_cv_01.yaml'
sleep 30

python src/train.py -c 'configs/experiment_002_128x128/config_test_03_cv_02.yaml'

echo "Finished!"