% uiopen('/Users/dineshk/Dropbox/Latex/sADMM/software/MATLAB implementation/Robot_classification/data/sensor_readings_4.csv',1)
class = grp2idx(sensorreadings4.SlightRightTurn);
data = [sensorreadings4.VarName1,...
    sensorreadings4.VarName2,...
    sensorreadings4.VarName2,...
    sensorreadings4.VarName2,...
    class];
