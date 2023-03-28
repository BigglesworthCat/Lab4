class SystemChecker:
    Y1_abnormal = 40
    Y1_a = 5

    Y2_abnormal = 100
    Y2_a = 50

    Y3_abnormal = 75
    Y3_a = 85
    
    Y4_abnormal = 80
    Y4_a = 100

    def __int__(self, Y):
        self.Y = Y

    def check(self, index):
        Y1 = self.Y.iloc[index, 1]
        Y2 = self.Y.iloc[index, 2]
        Y3 = self.Y.iloc[index, 3]
        Y4 = self.Y.iloc[index, 4]