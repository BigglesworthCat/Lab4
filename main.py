from tkinter import *
from tkinter.scrolledtext import ScrolledText

from additional import Mode, Form
from model import Model


class Application(Tk):
    def __init__(self):
        super().__init__()
        self.title('Лабораторна робота 4')
        self.resizable(False, False)

        # 'Режим'
        self.mode_label_frame = LabelFrame(self, text='Режим')
        self.mode_label_frame.grid(row=0, column=0, sticky='NWE', padx=5, pady=5, ipadx=5, ipady=5)

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
        self.form_label_frame.grid(row=0, column=1, sticky='NWE', padx=5, pady=5, ipadx=5, ipady=5)

        self.form = StringVar()
        self.form.set(Form.ADDITIVE.name)
        self.form_additive_radiobutton = Radiobutton(self.form_label_frame, text='Аддитивна', variable=self.form,
                                                     value=Form.ADDITIVE.name)
        self.form_additive_radiobutton.grid(row=0, sticky='W')
        self.form_multiplicative_radiobutton = Radiobutton(self.form_label_frame, text='Мультиплікативна',
                                                           variable=self.form, value=Form.MULTIPLICATIVE.name)
        self.form_multiplicative_radiobutton.grid(row=2, sticky='W')

        # 'Запустити'
        self.run_button = Button(self, text='Запустити',
                                 command=self.run,
                                 bg='red',
                                 fg='white'
                                 )
        self.run_button.grid(row=0, column=2, columnspan=2, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        # 'Графіки'
        self.plots_label_frame = LabelFrame(self, text='Графіки')
        self.plots_label_frame.grid(row=1, column=0, rowspan=4, columnspan=4, sticky='WE', padx=5, pady=5, ipadx=5,
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
        self.results_label_frame.grid(row=5, column=0, columnspan=4, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        self.result_area = ScrolledText(self.results_label_frame, height=5)
        self.result_area.pack(fill='both', expand=True)

    def check_status(self):
        pass

    def draw_point(self, plot, column, row):
        plot.delete('all')

        prev_x_point, prev_y_point, prev_y_pred_point = 0, plot.winfo_height(), plot.winfo_height()

        x_data = [i for i in range(row + 1)]
        y_data = self.Func.iloc[:row + 1, column].to_list()
        y_data_pred = self.Func_predicted.iloc[:row + 1, column].to_list()

        for x_point, y_point, y_pred_point in zip(x_data, y_data, y_data_pred):
            y_point = plot.winfo_height() - y_point
            y_pred_point = plot.winfo_height() - y_pred_point
            plot.create_line(prev_x_point, prev_y_point, x_point, y_point, fill='blue', width=2)
            plot.create_line(prev_x_point, prev_y_point, x_point, y_pred_point, fill='orange', width=2)
            prev_x_point = x_point
            prev_y_point = y_point
            prev_y_pred_point = prev_y_pred_point

    def update_values(self, index):
        self.Y1_value.set(self.Func_predicted.iloc[index, 1])
        self.Y2_value.set(self.Func_predicted.iloc[index, 2])
        self.Y3_value.set(self.Func_predicted.iloc[index, 3])
        self.Y4_value.set(self.Func_predicted.iloc[index, 4])

    def update_plots(self, index):
        self.draw_point(self.Y1_plot, 1, index)
        self.draw_point(self.Y2_plot, 2, index)
        self.draw_point(self.Y3_plot, 3, index)
        self.draw_point(self.Y4_plot, 4, index)

    def process_data(self, index):
        self.check_status()
        self.update_values(index)
        self.update_plots(index)
        if index != len(self.Func) - 1:
            self.after(1, self.process_data, index + 1)

    def run(self):
        model = Model(self.mode.get(), self.form.get())
        self.Func, self.Func_predicted = model.restore()
        self.after(1, self.process_data, 0)


if __name__ == "__main__":
    application = Application()
    application.mainloop()
