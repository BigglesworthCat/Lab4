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

        self.options_label_frame = LabelFrame(self, text='Налаштування')
        self.options_label_frame.grid(row=0, column=0, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        # 'Режим'
        self.mode_label_frame = LabelFrame(self.options_label_frame, text='Режим')
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
        self.form_label_frame = LabelFrame(self.options_label_frame, text='Форма')
        self.form_label_frame.grid(row=1, column=0, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        self.form = StringVar()
        self.form.set(Form.ADDITIVE.name)
        self.form_additive_radiobutton = Radiobutton(self.form_label_frame, text='Аддитивна', variable=self.form,
                                                     value=Form.ADDITIVE.name)
        self.form_additive_radiobutton.grid(row=0, sticky='W')
        self.form_multiplicative_radiobutton = Radiobutton(self.form_label_frame, text='Мультиплікативна',
                                                           variable=self.form, value=Form.MULTIPLICATIVE.name)
        self.form_multiplicative_radiobutton.grid(row=2, sticky='W')

        self.shifts_label_frame = LabelFrame(self.options_label_frame, text='Зсуви')
        self.shifts_label_frame.grid(row=3, column=0, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        self.N_02_value = StringVar()
        self.N_02_label = Label(self.shifts_label_frame, text='N_02:')
        self.N_02_label.grid(row=0, column=0, sticky='E')
        self.N_02_spinbox = Spinbox(self.shifts_label_frame, from_=1, to=100, width=5, textvariable=self.N_02_value)
        self.N_02_spinbox.grid(row=0, column=1, sticky='WE', padx=5, pady=2)
        self.N_02_value.set('40')

        self.prediction_step_value = StringVar()
        self.prediction_step_label = Label(self.shifts_label_frame, text='Крок передбачення:')
        self.prediction_step_label.grid(row=1, column=0, sticky='E')
        self.prediction_step_spinbox = Spinbox(self.shifts_label_frame, from_=1, to=100, width=5,
                                               textvariable=self.prediction_step_value)
        self.prediction_step_spinbox.grid(row=1, column=1, sticky='WE', padx=5, pady=2)
        self.prediction_step_value.set('20')

        # 'Поліноми'
        self.polynomials_label_frame = LabelFrame(self, text='Поліноми')
        self.polynomials_label_frame.grid(row=0, column=1, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

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

        self.p1_value = StringVar()
        self.p1_label = Label(self.polynomials_dimensions_label_frame, text='P1:')
        self.p1_label.grid(row=0, column=0, sticky='E')
        self.p1_spinbox = Spinbox(self.polynomials_dimensions_label_frame, from_=1, to=4, width=5,
                                  textvariable=self.p1_value)
        self.p1_spinbox.grid(row=0, column=1, sticky='WE', padx=5, pady=2)
        self.p1_value.set('3')

        self.p2_value = StringVar()
        self.p2_label = Label(self.polynomials_dimensions_label_frame, text='P2:')
        self.p2_label.grid(row=1, column=0, sticky='E')
        self.p2_spinbox = Spinbox(self.polynomials_dimensions_label_frame, from_=1, to=4, width=5,
                                  textvariable=self.p2_value)
        self.p2_spinbox.grid(row=1, column=1, sticky='WE', padx=5, pady=2)
        self.p2_value.set('3')

        self.p3_value = StringVar()
        self.p3_label = Label(self.polynomials_dimensions_label_frame, text='P3:')
        self.p3_label.grid(row=2, column=0, sticky='E')
        self.p3_spinbox = Spinbox(self.polynomials_dimensions_label_frame, from_=1, to=4, width=5,
                                  textvariable=self.p3_value)
        self.p3_spinbox.grid(row=2, column=1, sticky='WE', padx=5, pady=2)
        self.p3_value.set('3')

        self.p4_value = StringVar()
        self.p4_label = Label(self.polynomials_dimensions_label_frame, text='P4:')
        self.p4_label.grid(row=3, column=0, sticky='E')
        self.p4_spinbox = Spinbox(self.polynomials_dimensions_label_frame, from_=1, to=4, width=5,
                                  textvariable=self.p4_value)
        self.p4_spinbox.grid(row=3, column=1, sticky='WE', padx=5, pady=2)
        self.p4_value.set('3')

        # 'Додатково'
        self.additional_label_frame = LabelFrame(self, text='Додатково')
        self.additional_label_frame.grid(row=0, column=2, sticky='WENS', padx=5, pady=5,
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

        self.normalization_label_frame = LabelFrame(self.additional_label_frame, text='Нормалізація')

        self.normalization = StringVar()
        self.normalization.set(Normalization.NORMED.name)
        self.normalization_label_frame.grid(row=0, column=1, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)
        self.normed_graph_radiobutton = Radiobutton(self.normalization_label_frame, text='Нормалізовані графіки',
                                                    variable=self.normalization, value=Normalization.NORMED.name)
        self.normed_graph_radiobutton.grid(row=0, sticky='W')
        self.unnormed_graph_radiobutton = Radiobutton(self.normalization_label_frame, text='Не нормалізовані графіки',
                                                    variable=self.normalization, value=Normalization.UNNORMED.name)
        self.unnormed_graph_radiobutton.grid(row=1, sticky='W')

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
        self.plots_label_frame.grid(row=3, column=0, rowspan=4, columnspan=4, sticky='WE', padx=5, pady=5, ipadx=5,
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

        self.Y1_plot = Canvas(self.Y1_label_frame, width=PLOT_WIDTH, height=PLOT_HEIGHT, bg='white')
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

        self.Y2_plot = Canvas(self.Y2_label_frame, width=PLOT_WIDTH, height=PLOT_HEIGHT, bg='white')
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

        self.Y3_plot = Canvas(self.Y3_label_frame, width=PLOT_WIDTH, height=PLOT_HEIGHT, bg='white')
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

        self.Y4_plot = Canvas(self.Y4_label_frame, width=PLOT_WIDTH, height=PLOT_HEIGHT, bg='white')
        self.Y4_plot.grid(row=1, column=0, columnspan=2, sticky='WE', padx=5, pady=2)

        # 'Результати'
        self.results_label_frame = LabelFrame(self, text='Результати')
        self.results_label_frame.grid(row=7, column=0, columnspan=4, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        self.result_area = ScrolledText(self.results_label_frame, height=10)
        self.result_area.pack(fill='both', expand=True)

        # shift all plots by 5 pixels up
        self.shift_y = -5

    def draw_point(self, plot, column, row, limits):
        plot.delete('all')
        prev_x_point, prev_y_point, prev_y_pred_point = 0, plot.winfo_height(), plot.winfo_height()
        plot.create_line(0, plot.winfo_height() - limits[0] + self.shift_y, plot.winfo_width(),
                         plot.winfo_height() - limits[0] + self.shift_y,
                         fill='pink', width=2)
        plot.create_line(0, plot.winfo_height() - limits[1] + self.shift_y, plot.winfo_width(),
                         plot.winfo_height() - limits[1] + self.shift_y,
                         fill='green', width=2)
        plot.create_line(500, self.winfo_height(), 500, 0, fill='purple')
        x_data = [i for i in range(row + 1)]
        y_data = self.Y.iloc[:row + 1, column].to_list()
        y_data_pred = self.Y_pred.iloc[:row + 1, column].to_list()

        for x_point, y_point in zip(x_data, y_data):
            y_point = plot.winfo_height() - y_point + self.shift_y
            plot.create_line(prev_x_point, prev_y_point, x_point, y_point, fill='blue', width=2)
            prev_x_point = x_point
            prev_y_point = y_point

        prev_x_point, prev_y_point, prev_y_pred_point = 0, plot.winfo_height(), plot.winfo_height()
        for x_point, y_pred_point in zip(x_data, y_data_pred):
            y_pred_point = plot.winfo_height() - y_pred_point + self.shift_y
            plot.create_line(prev_x_point, prev_y_point, x_point, y_pred_point, fill='orange', width=2)
            prev_x_point = x_point
            prev_y_point = y_pred_point

        # Define the coordinates of the X-axis line
        x1, y1 = 0, plot.winfo_height()
        x2, y2 = plot.winfo_width(), plot.winfo_height()
        ticks_range = 40
        ticks = x2 // ticks_range
        # Draw the tick marks and labels
        for i in range(1, ticks):
            x = i * ticks_range
            plot.create_line(x, y1, x, y1 - 5)
            plot.create_text(x, y1 - 10, text=str(i * ticks_range * 20), fill="black", font=("Arial", 5))

        ticks = y1 // ticks_range
        if self.normalization.get() == Normalization.NORMED.name:
            y_tick_values = [round((i * ticks_range + ticks_range) / max(self.Y.iloc[:, column].to_list()), 3) for i in range(ticks)]
        else:
            y_tick_values = [i * ticks_range + ticks_range for i in range(ticks)]
        y_tick_values.reverse()
        if len(y_tick_values) != ticks:
            print(ticks)
            print(y_tick_values)
            raise ValueError
        for j in range(ticks):
            y = j * ticks_range
            plot.create_line(5, y + self.shift_y, 0, y + self.shift_y)
            plot.create_text(10, y + self.shift_y, text=str(y_tick_values[j]), fill="black", font=("Arial", 5))

    def update_values(self, index):
        self.Y1_value.set(self.Y_pred.iloc[index, 1])
        self.Y2_value.set(self.Y_pred.iloc[index, 2])
        self.Y3_value.set(self.Y_pred.iloc[index, 3])
        self.Y4_value.set(self.Y_pred.iloc[index, 4])

    def update_plots(self, index):
        self.draw_point(self.Y1_plot, 1, index, self.limits[0])
        self.draw_point(self.Y2_plot, 2, index, self.limits[1])
        self.draw_point(self.Y3_plot, 3, index, self.limits[2])
        self.draw_point(self.Y4_plot, 4, index, self.limits[3])

    def update_status(self, index):
        status = self.checker.get_status(index)
        danger_level = status['danger_level']
        situation_type = status['situation_type']
        situation_description = status['situation_description']
        time_before_a = status['time_before_a']
        self.result_area.insert(END,
                                f'Time: {(index + 1) * 20}\n Available risk : {time_before_a} \n Danger level: {danger_level}\nSituation type: {situation_type}\n')
        if situation_description != '':
            self.result_area.insert(END, f'Situation description: {situation_description}\n\n')
        else:
            self.result_area.insert(END, '\n')

    def process_data(self, index):
        self.update_values(index)
        self.update_plots(index)
        self.update_status(index)
        if index != len(self.Y_pred) - 1:
            self.after(1, self.process_data, index + 1)

    def run(self):
        self.result_area.delete('1.0', END)
        P_dims = [int(self.p1_spinbox.get()), int(self.p2_spinbox.get()), int(self.p3_spinbox.get())]
        model = Model(self.mode.get(), self.form.get(), int(self.N_02_spinbox.get()),
                      int(self.prediction_step_spinbox.get()), self.polynomial_var.get(), P_dims, self.weight.get(),
                      self.lambdas.get())
        self.Y, self.Y_pred = model.restore()
        print(self.Y_pred.shape)
        self.checker = SystemChecker(self.Y_pred)
        self.limits = self.checker.get_bounds()

        self.after(1, self.process_data, 0)


if __name__ == "__main__":
    application = Application()
    application.mainloop()
