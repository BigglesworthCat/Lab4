from tkinter import *
from tkinter.scrolledtext import ScrolledText

from additional import Mode, Form
from model import Model

window = Tk()
window.title('Лабораторна робота 4')
window.resizable(False, False)


class Application:
    def __init__(self, window):
        # 'Режим'
        self.mode_label_frame = LabelFrame(window, text='Режим')
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
        self.form_label_frame = LabelFrame(window, text='Форма')
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
        self.run_button = Button(window, text='Запустити',
                                 command=self.run,
                                 bg='red',
                                 fg='white'
                                 )
        self.run_button.grid(row=0, column=2, columnspan=2, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        # 'Графіки'
        self.plots_label_frame = LabelFrame(window, text='Графіки')
        self.plots_label_frame.grid(row=1, column=0, rowspan=4, columnspan=4, sticky='WE', padx=5, pady=5, ipadx=5,
                                    ipady=5)

        # 'Y1'
        self.Y1_label_frame = LabelFrame(self.plots_label_frame, text='Y1')
        self.Y1_label_frame.grid(row=0, column=0, rowspan=2, columnspan=2, sticky='WE')

        self.Y1_label = Label(self.Y1_label_frame, text='Y1:')
        self.Y1_label.grid(row=0, column=0, sticky='WE', padx=5, pady=2)

        self.Y1_value = StringVar()
        self.Y1_value.set('')
        self.Y1_entry = Entry(self.Y1_label_frame, textvariable=self.Y1_value, state=DISABLED)
        self.Y1_entry.grid(row=0, column=1, sticky='WE', padx=5, pady=2)

        self.Y1_plot = Canvas(self.Y1_label_frame)
        self.Y1_plot.grid(row=1, column=0, columnspan=2, sticky='WE', padx=5, pady=2)

        # 'Y2'
        self.Y2_label_frame = LabelFrame(self.plots_label_frame, text='Y2')
        self.Y2_label_frame.grid(row=0, column=2, rowspan=2, columnspan=2, sticky='WE')

        self.Y2_label = Label(self.Y2_label_frame, text='Y2:')
        self.Y2_label.grid(row=0, column=0, sticky='WE', padx=5, pady=2)

        self.Y2_value = StringVar()
        self.Y2_value.set('')
        self.Y2_entry = Entry(self.Y2_label_frame, textvariable=self.Y2_value, state=DISABLED)
        self.Y2_entry.grid(row=0, column=1, sticky='WE', padx=5, pady=2)

        self.Y2_plot = Canvas(self.Y2_label_frame)
        self.Y2_plot.grid(row=1, column=0, columnspan=2, sticky='WE', padx=5, pady=2)

        # 'Y3'
        self.Y3_label_frame = LabelFrame(self.plots_label_frame, text='Y3')
        self.Y3_label_frame.grid(row=2, column=0, rowspan=2, columnspan=2, sticky='WE')

        self.Y3_label = Label(self.Y3_label_frame, text='Y3:')
        self.Y3_label.grid(row=0, column=0, sticky='WE', padx=5, pady=2)

        self.Y3_value = StringVar()
        self.Y3_value.set('')
        self.Y3_entry = Entry(self.Y3_label_frame, textvariable=self.Y3_value, state=DISABLED)
        self.Y3_entry.grid(row=0, column=1, sticky='WE', padx=5, pady=2)

        self.Y3_plot = Canvas(self.Y3_label_frame)
        self.Y3_plot.grid(row=1, column=0, columnspan=2, sticky='WE', padx=5, pady=2)

        # 'Y4'
        self.Y4_label_frame = LabelFrame(self.plots_label_frame, text='Y4')
        self.Y4_label_frame.grid(row=2, column=2, rowspan=2, columnspan=2, sticky='WE')

        self.Y4_label = Label(self.Y4_label_frame, text='Y4:')
        self.Y4_label.grid(row=0, column=0, sticky='WE', padx=5, pady=2)

        self.Y4_value = StringVar()
        self.Y4_value.set('')
        self.Y4_entry = Entry(self.Y4_label_frame, textvariable=self.Y4_value, state=DISABLED)
        self.Y4_entry.grid(row=0, column=1, sticky='WE', padx=5, pady=2)

        self.Y4_plot = Canvas(self.Y4_label_frame)
        self.Y4_plot.grid(row=1, column=0, columnspan=2, sticky='WE', padx=5, pady=2)

        # 'Результати'
        self.results_label_frame = LabelFrame(window, text='Результати')
        self.results_label_frame.grid(row=5, column=0, columnspan=4, sticky='WENS', padx=5, pady=5, ipadx=5, ipady=5)

        self.result_area = ScrolledText(self.results_label_frame, height=5)
        self.result_area.pack(fill='both', expand=True)

    def process_data(self):
        for i in range(len(self.Func)):
            print(i)

    def run(self):
        model = Model(self.mode.get(), self.form.get())
        self.Func, self.Func_predicted = model.restore()
        self.process_data()


application = Application(window)

window.mainloop()
