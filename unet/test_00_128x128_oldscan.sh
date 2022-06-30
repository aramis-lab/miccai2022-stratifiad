echo "Starting training for test 00 in cuda:0"

python src/train.py -c 'configs/experiment_005_128x128/config_test_00_cv_00.yaml'
sleep 30

python src/train.py -c 'configs/experiment_005_128x128/config_test_00_cv_01.yaml'
sleep 30

python src/train.py -c 'configs/experiment_005_128x128/config_test_00_cv_02.yaml'
sleep 30

python src/train.py -c 'configs/experiment_005_128x128/config_test_00_cv_03.yaml'

echo "Finished!"