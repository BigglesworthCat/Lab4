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

    def get_risk_rate(self, index):
        ## return ступінь ризику
        Y1 = self.Y.iloc[index, 1]
        Y2 = self.Y.iloc[index, 2]
        Y3 = self.Y.iloc[index, 3]
        Y4 = self.Y.iloc[index, 4]
        reason = []

        Py1,Py2,Py3,Py4 = 0,0,0,0
        if Y1 < self.Y1_abnormal:
            if Y1 < self.Y1_a:
                Py1 = 1
                reason.append('Рівень води в першому резервуарі аварійно малий')
            else:
                reason.append('Рівень води в першому резервуарі нештатний')
                Py1 = abs(Y1 - self.Y1_abnormal)/abs(self.Y1_abnormal- self.Y1_a)
        
        if Y2 < self.Y2_abnormal:
            if Y2 < self.Y2_a:
                Py2 = 1
                reason.append('Напір аварійно низький')
                
            else:
                reason.append('Напір нештатний')
                Py2 = abs(Y2 - self.Y2_abnormal)/abs(self.Y2_abnormal- self.Y2_a)


        if Y3 > self.Y3_abnormal:
            if Y3 > self.Y3_a:
                reason.append('Температура аварійно висока')
                Py3 = 1
            else:
                reason.append('Температура нештатна')
                Py3 = abs(Y3 - self.Y3_abnormal)/abs(self.Y3_abnormal- self.Y3_a)
                
        if Y4 > self.Y4_abnormal:
            if Y4 > self.Y4_a:
                reason.append('Рівень води в другому резервуарі аварійно високий')
                Py4 = 1
            else:
                reason.append('Рівень води в другому резервуарі нештатний')
                
                Py4 = abs(Y4 - self.Y4_abnormal)/abs(self.Y4_abnormal- self.Y4_a)

        self.F =  1 - (1-Py1)*(1-Py2)*(1-Py3)*(1-Py4)
        self.reasons = reason








    def get_status(self,index):
        self.get_risk_rate(index)
        F = self.F
        reasons = self.reasons
        if 0 <= F <= 1/8: 
            description = 'Безпечна ситуація '
        elif F <=1/4:
            description = 'Позаштатна ситуація за одним параметром'
        elif F <= 3/8:
            description = 'Позаштатна ситуація за декількома параметрами'
        elif F <= 1/2:
            description = 'Спостерігається загроза аварії'
        elif F <= 5/8:
            description = 'Висока загроза аварії'
        elif F <= 3/4:
            description = 'Критична ситуація'
        elif F <= 7/8:
            description = 'Шанс уникнути аварії винятково малий'
        elif F <= 1:
            description = 'Аварія'

        return {'level_of_danger': F, 'description': description, 'reasons': reasons}

            
        