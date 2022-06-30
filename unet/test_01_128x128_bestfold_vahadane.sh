echo "Starting training for test 01 in cuda:0"

#python src/train.py -c 'configs/experiment_002_128x128/config_test_01_cv_00.yaml'
#sleep 30

python src/train.py -c 'configs/experiment_006_128x128/config_test_01_cv_01.yaml'
#sleep 30

#python src/train.py -c 'configs/experiment_002_128x128/config_test_01_cv_02.yaml'

echo "Finished!"