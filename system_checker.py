class SystemChecker:
    Y1_abnormal = 40
    Y1_a = 5

    Y2_abnormal = 100
    Y2_a = 50

    Y3_abnormal = 75
    Y3_a = 85

    Y4_abnormal = 80
    Y4_a = 100

    def __init__(self, Y):
        self.Y = Y

    def get_abnormal_and_bebra(self):
        return [[self.Y1_a, self.Y1_abnormal], [self.Y2_a, self.Y2_abnormal],\
            [self.Y3_a, self.Y3_abnormal], [self.Y4_a, self.Y4_abnormal]]

    def get_risk_rate_and_description(self, index):
        Y1 = self.Y.iloc[index, 1]
        Y2 = self.Y.iloc[index, 2]
        Y3 = self.Y.iloc[index, 3]
        Y4 = self.Y.iloc[index, 4]
        situation_description = []

        P_y1, P_y2, P_y3, P_y4 = 0, 0, 0, 0
        if Y1 < self.Y1_abnormal:
            if Y1 < self.Y1_a:
                P_y1 = 1
                situation_description.append('Рівень води в першому резервуарі аварійно малий.')
            else:
                situation_description.append('Рівень води в першому резервуарі нештатний.')
                P_y1 = abs(Y1 - self.Y1_abnormal) / abs(self.Y1_abnormal - self.Y1_a)

        if Y2 < self.Y2_abnormal:
            if Y2 < self.Y2_a:
                P_y2 = 1
                situation_description.append('Напір аварійно низький.')
            else:
                situation_description.append('Напір нештатний.')
                P_y2 = abs(Y2 - self.Y2_abnormal) / abs(self.Y2_abnormal - self.Y2_a)

        if Y3 > self.Y3_abnormal:
            if Y3 > self.Y3_a:
                situation_description.append('Температура аварійно висока.')
                P_y3 = 1
            else:
                situation_description.append('Температура нештатна.')
                P_y3 = abs(Y3 - self.Y3_abnormal) / abs(self.Y3_abnormal - self.Y3_a)

        if Y4 > self.Y4_abnormal:
            if Y4 > self.Y4_a:
                situation_description.append('Рівень води в другому резервуарі аварійно високий.')
                P_y4 = 1
            else:
                situation_description.append('Рівень води в другому резервуарі нештатний.')
                P_y4 = abs(Y4 - self.Y4_abnormal) / abs(self.Y4_abnormal - self.Y4_a)

        self.F = 1 - (1 - P_y1) * (1 - P_y2) * (1 - P_y3) * (1 - P_y4)
        self.situation_description = "\n".join(situation_description)

    def get_situation_type(self):
        if 0 <= self.F <= 1 / 8:
            self.situation_type = 'Безпечна ситуація.'
        elif self.F <= 1 / 4:
            self.situation_type = 'Позаштатна ситуація за одним параметром.'
        elif self.F <= 3 / 8:
            self.situation_type = 'Позаштатна ситуація за декількома параметрами.'
        elif self.F <= 1 / 2:
            self.situation_type = 'Спостерігається загроза аварії.'
        elif self.F <= 5 / 8:
            self.situation_type = 'Висока загроза аварії.'
        elif self.F <= 3 / 4:
            self.situation_type = 'Критична ситуація.'
        elif self.F <= 7 / 8:
            self.situation_type = 'Шанс уникнути аварії винятково малий.'
        elif self.F <= 1:
            self.situation_type = 'Аварія.'

    def get_status(self, index):
        self.get_risk_rate_and_description(index)
        self.get_situation_type()

        return {'danger_level': self.F, 'situation_type': self.situation_type,
                'situation_description': self.situation_description}
