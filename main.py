from tkinter import *
from tkinter.scrolledtext import ScrolledText

from additional import *
from model import Model
from system_checker import SystemChecker


class Application(Tk):
    def __init__(self):
        super().__init__()
        self.title('Лабораторна робота 4')
        # self.resizable(False, False)

        # 'Режим'
        self.mode_label_frame = LabelFrame(self, text='Режим')
        self.mode_label_frame.grid(row=0, column=0, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        self.mode = StringVar()
        self.mode.set(Mode.NORMAL.name)
        self.mode_normal_radiobutton = Radiobutton(self.mode_label_frame, text='Нормальний', variable=self.mode,
                                                   value=Mode.NORMAL.name)
        self.mode_normal_radiobutton.grid(row=0, sticky='W')
        self.mode_extraordinary_radiobutton = Radiobutton(self.mode_label_frame, text='Позаштатний', variable=self.mode,
                                                          value=Mode.EXTRAORDINARY.name)
        self.mode_extraordinary_radiobutton.grid(row=1, sticky='W')

        # 'Форма наближувальної функції'
        self.form_label_frame = LabelFrame(self, text='Форма')
        self.form_label_frame.grid(row=1, column=0, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        self.form = StringVar()
        self.form.set(Form.ADDITIVE.name)
        self.form_additive_radiobutton = Radiobutton(self.form_label_frame, text='Аддитивна', variable=self.form,
                                                     value=Form.ADDITIVE.name)
        self.form_additive_radiobutton.grid(row=0, sticky='W')
        self.form_multiplicative_radiobutton = Radiobutton(self.form_label_frame, text='Мультиплікативна',
                                                           variable=self.form, value=Form.MULTIPLICATIVE.name)
        self.form_multiplicative_radiobutton.grid(row=2, sticky='W')

        # 'Поліноми'
        self.polynomials_label_frame = LabelFrame(self, text='Поліноми')
        self.polynomials_label_frame.grid(row=0, column=1, rowspan=2, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        ## 'Вид поліномів'
        self.polynomials_type_label_frame = LabelFrame(self.polynomials_label_frame, text='Вид поліномів')
        self.polynomials_type_label_frame.grid(row=0, column=0, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        self.polynomial_var = StringVar()
        self.polynomial_var.set(Polynom.CHEBYSHEV.name)
        self.chebyshev_radiobutton = Radiobutton(self.polynomials_type_label_frame, text='Поліноми Чебишева',
                                                 variable=self.polynomial_var,
                                                 value=Polynom.CHEBYSHEV.name)
        self.chebyshev_radiobutton.grid(row=0, sticky='W')
        self.legandre_radiobutton = Radiobutton(self.polynomials_type_label_frame, text='Поліноми Лежандра',
                                                variable=self.polynomial_var,
                                                value=Polynom.LEGANDRE.name)
        self.legandre_radiobutton.grid(row=1, sticky='W')
        self.lagerr_radiobutton = Radiobutton(self.polynomials_type_label_frame, text='Поліноми Лагерра',
                                              variable=self.polynomial_var,
                                              value=Polynom.LAGERR.name)
        self.lagerr_radiobutton.grid(row=2, sticky='W')

        ## 'Степені поліномів'
        self.polynomials_dimensions_label_frame = LabelFrame(self.polynomials_label_frame, text='Степені поліномів')
        self.polynomials_dimensions_label_frame.grid(row=1, column=0, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        self.polynomials_dimensions_label_frame.columnconfigure(1, weight=1)
        self.p1_label = Label(self.polynomials_dimensions_label_frame, text='P1:')
        self.p1_label.grid(row=0, column=0, sticky='E')
        self.p1_spinbox = Spinbox(self.polynomials_dimensions_label_frame, from_=1, to=4, width=5)
        self.p1_spinbox.grid(row=0, column=1, sticky='WE', padx=5, pady=2)

        self.p2_label = Label(self.polynomials_dimensions_label_frame, text='P2:')
        self.p2_label.grid(row=1, column=0, sticky='E')
        self.p2_spinbox = Spinbox(self.polynomials_dimensions_label_frame, from_=1, to=4, width=5)
        self.p2_spinbox.grid(row=1, column=1, sticky='WE', padx=5, pady=2)

        self.p3_label = Label(self.polynomials_dimensions_label_frame, text='P3:')
        self.p3_label.grid(row=2, column=0, sticky='E')
        self.p3_spinbox = Spinbox(self.polynomials_dimensions_label_frame, from_=1, to=4, width=5)
        self.p3_spinbox.grid(row=2, column=1, sticky='WE', padx=5, pady=2)

        # 'Додатково'
        self.additional_label_frame = LabelFrame(self, text='Додатково')
        self.additional_label_frame.grid(row=0, column=2, rowspan=2, columnspan=2, sticky='WENS', padx=5, pady=5,
                                         ipadx=5, ipady=5)

        ## 'Ваги цільових функцій'
        self.weight_label_frame = LabelFrame(self.additional_label_frame, text='Ваги цільових функцій')
        self.weight_label_frame.grid(row=0, column=0, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        self.weight = StringVar()
        self.weight.set(Weight.NORMED.name)
        self.normed_radiobutton = Radiobutton(self.weight_label_frame, text='Нормовані Yi', variable=self.weight,
                                              value=Weight.NORMED.name)
        self.normed_radiobutton.grid(row=0, sticky='W')
        self.min_max_radiobutton = Radiobutton(self.weight_label_frame, text='(min(Yi) + max(Yi)) / 2',
                                               variable=self.weight,
                                               value=Weight.MIN_MAX.name)
        self.min_max_radiobutton.grid(row=1, sticky='W')

        ## 'Метод визначення лямбд'
        self.lambdas_label_frame = LabelFrame(self.additional_label_frame, text='Метод визначення лямбд')
        self.lambdas_label_frame.grid(row=1, column=0, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        self.lambdas = StringVar()
        self.lambdas.set(Lambda.SINGLE_SET.name)
        self.single_set_radiobutton = Radiobutton(self.lambdas_label_frame, text='Одна система', variable=self.lambdas,
                                                  value=Lambda.SINGLE_SET.name)
        self.single_set_radiobutton.grid(row=0, sticky='W')
        self.triple_set_radiobutton = Radiobutton(self.lambdas_label_frame, text='Три системи', variable=self.lambdas,
                                                  value=Lambda.TRIPLE_SET.name)
        self.triple_set_radiobutton.grid(row=1, sticky='W')

        ## 'Запустити'
        self.run_button = Button(self.additional_label_frame, text='Запустити',
                                 command=self.run,
                                 bg='red',
                                 fg='white'
                                 )
        self.run_button.grid(row=2, column=0, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        # 'Графіки'
        self.plots_label_frame = LabelFrame(self, text='Графіки')
        self.plots_label_frame.grid(row=2, column=0, rowspan=4, columnspan=4, sticky='WE', padx=5, pady=5, ipadx=5,
                                    ipady=5)

        # 'Y1'
        self.Y1_label_frame = LabelFrame(self.plots_label_frame, text='Y1')
        self.Y1_label_frame.grid(row=0, column=0, rowspan=2, columnspan=2, sticky='WE')

        self.Y1_label = Label(self.Y1_label_frame, text='Y1:')
        self.Y1_label.grid(row=0, column=0, sticky='E', padx=5, pady=2)

        self.Y1_value = StringVar()
        self.Y1_value.set('')
        self.Y1_entry = Entry(self.Y1_label_frame, textvariable=self.Y1_value, state=DISABLED,
                              disabledbackground='white', disabledforeground='black')
        self.Y1_entry.grid(row=0, column=1, sticky='W', padx=5, pady=2)

        self.Y1_plot = Canvas(self.Y1_label_frame, width=500, height=200, bg='white')
        self.Y1_plot.grid(row=1, column=0, columnspan=2, sticky='WE', padx=5, pady=2)

        # 'Y2'
        self.Y2_label_frame = LabelFrame(self.plots_label_frame, text='Y2')
        self.Y2_label_frame.grid(row=0, column=2, rowspan=2, columnspan=2, sticky='WE')

        self.Y2_label = Label(self.Y2_label_frame, text='Y2:')
        self.Y2_label.grid(row=0, column=0, sticky='E', padx=5, pady=2)

        self.Y2_value = StringVar()
        self.Y2_value.set('')
        self.Y2_entry = Entry(self.Y2_label_frame, textvariable=self.Y2_value, state=DISABLED,
                              disabledbackground='white', disabledforeground='black')
        self.Y2_entry.grid(row=0, column=1, sticky='W', padx=5, pady=2)

        self.Y2_plot = Canvas(self.Y2_label_frame, width=500, height=200, bg='white')
        self.Y2_plot.grid(row=1, column=0, columnspan=2, sticky='WE', padx=5, pady=2)

        # 'Y3'
        self.Y3_label_frame = LabelFrame(self.plots_label_frame, text='Y3')
        self.Y3_label_frame.grid(row=2, column=0, rowspan=2, columnspan=2, sticky='WE')

        self.Y3_label = Label(self.Y3_label_frame, text='Y3:')
        self.Y3_label.grid(row=0, column=0, sticky='E', padx=5, pady=2)

        self.Y3_value = StringVar()
        self.Y3_value.set('')
        self.Y3_entry = Entry(self.Y3_label_frame, textvariable=self.Y3_value, state=DISABLED,
                              disabledbackground='white', disabledforeground='black')
        self.Y3_entry.grid(row=0, column=1, sticky='W', padx=5, pady=2)

        self.Y3_plot = Canvas(self.Y3_label_frame, width=500, height=200, bg='white')
        self.Y3_plot.grid(row=1, column=0, columnspan=2, sticky='WE', padx=5, pady=2)

        # 'Y4'
        self.Y4_label_frame = LabelFrame(self.plots_label_frame, text='Y4')
        self.Y4_label_frame.grid(row=2, column=2, rowspan=2, columnspan=2, sticky='WE')

        self.Y4_label = Label(self.Y4_label_frame, text='Y4:')
        self.Y4_label.grid(row=0, column=0, sticky='E', padx=5, pady=2)

        self.Y4_value = StringVar()
        self.Y4_value.set('')
        self.Y4_entry = Entry(self.Y4_label_frame, textvariable=self.Y4_value, state=DISABLED,
                              disabledbackground='white', disabledforeground='black')
        self.Y4_entry.grid(row=0, column=1, sticky='W', padx=5, pady=2)

        self.Y4_plot = Canvas(self.Y4_label_frame, width=500, height=200, bg='white')
        self.Y4_plot.grid(row=1, column=0, columnspan=2, sticky='WE', padx=5, pady=2)

        # 'Результати'
        self.results_label_frame = LabelFrame(self, text='Результати')
        self.results_label_frame.grid(row=6, column=0, columnspan=4, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        self.result_area = ScrolledText(self.results_label_frame, height=5)
        self.result_area.pack(fill='both', expand=True)

    def check_status(self):
        pass

    def draw_point(self, plot, column, row):
        plot.delete('all')

        prev_x_point, prev_y_point, prev_y_pred_point = 0, plot.winfo_height(), plot.winfo_height()

        x_data = [i for i in range(row + 1)]
        y_data = self.Y.iloc[:row + 1, column].to_list()
        y_data_pred = self.Y_pred.iloc[:row + 1, column].to_list()

        for x_point, y_point, y_pred_point in zip(x_data, y_data, y_data_pred):
            y_point = plot.winfo_height() - y_point
            y_pred_point = plot.winfo_height() - y_pred_point
            plot.create_line(prev_x_point, prev_y_point, x_point, y_point, fill='blue', width=2)
            plot.create_line(prev_x_point, prev_y_point, x_point, y_pred_point, fill='orange', width=2)
            prev_x_point = x_point
            prev_y_point = y_point
            prev_y_pred_point = prev_y_pred_point

    def update_values(self, index):
        self.Y1_value.set(self.Y_pred.iloc[index, 1])
        self.Y2_value.set(self.Y_pred.iloc[index, 2])
        self.Y3_value.set(self.Y_pred.iloc[index, 3])
        self.Y4_value.set(self.Y_pred.iloc[index, 4])

    def update_plots(self, index):
        self.draw_point(self.Y1_plot, 1, index)
        self.draw_point(self.Y2_plot, 2, index)
        self.draw_point(self.Y3_plot, 3, index)
        self.draw_point(self.Y4_plot, 4, index)

    def update_status(self, index):
        status = self.checker.get_status(index)
        danger_level = status['danger_level']
        situation_type = status['situation_type']
        situation_description = status['situation_description']
        self.result_area.insert(END, f'Danger level: {danger_level}\nSituation type: {situation_type}\n')
        if situation_description != '':
            self.result_area.insert(END, f'Situation description: {situation_description}\n\n')
        else:
            self.result_area.insert(END, '\n')

    def process_data(self, index):
        self.check_status()
        self.update_values(index)
        self.update_plots(index)
        self.update_status(index)
        if index != len(self.Y) - 1:
            self.after(1, self.process_data, index + 1)

    def run(self):
        self.result_area.delete('1.0', END)
        model = Model(self.mode.get(), self.form.get())
        self.Y, self.Y_pred = model.restore_linear()
        self.checker = SystemChecker(self.Y_pred)

        self.after(1, self.process_data, 0)


if __name__ == "__main__":
    application = Application()
    application.mainloop()
