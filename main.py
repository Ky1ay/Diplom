import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout, QLabel, QLineEdit, \
    QPushButton, QTabWidget, QMessageBox, QComboBox, QTextEdit, QGroupBox, QHBoxLayout, QFileDialog
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sympy
import scipy.integrate
import mpmath



class TransformerAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ трансформатора тока")
        self.setGeometry(100, 100, 800, 600)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.input_tab = QWidget()
        self.result_tab = QWidget()
        self.graph_tab = QWidget()
        self.graph_anm_tab = QWidget()

        self.tabs.addTab(self.input_tab, "Ввод исходных данных")
        self.tabs.addTab(self.result_tab, "Результаты расчёта")
        self.tabs.addTab(self.graph_tab, "Графики первичного и вторичного тока")
        self.tabs.addTab(self.graph_anm_tab, "Характеристика намагничивания")

        self.init_input_tab()
        self.init_result_tab()
        self.init_graph_tab()
        self.init_graph_anm_tab()

    def init_input_tab(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Данные ТТ
        tt_group_box = QGroupBox("Данные ТТ")
        tt_layout = QFormLayout()
        self.voltage_edit = QLineEdit()
        self.frequency_combobox = QComboBox()
        self.primary_current_edit = QLineEdit()
        self.secondary_current_combobox = QComboBox()
        self.secondary_load_edit = QLineEdit()
        self.power_factor_edit = QLineEdit()
        self.max_k_factor_edit = QLineEdit()
        self.excitation_current_edit = QLineEdit()
        self.secondary_winding_turns_edit = QLineEdit()
        self.secondary_winding_active_resistance_edit = QLineEdit()
        self.secondary_winding_reactive_resistance_edit = QLineEdit()
        self.secondary_winding_inductance_edit = QLineEdit()
        self.remanence_coef_edit = QLineEdit()
        self.cross_magnet_steel_edit = QLineEdit()
        self.aver_leng_magn_field_line_edit = QLineEdit()

        tt_layout.addRow("Номинальное напряжение, кВ", self.voltage_edit)
        self.frequency_combobox.addItems(["50", "60"])
        tt_layout.addRow("Номинальная частота, Гц", self.frequency_combobox)
        tt_layout.addRow("Номинальный первичный ток, А", self.primary_current_edit)
        self.secondary_current_combobox.addItems(["1", "5"])
        tt_layout.addRow("Номинальный вторичный ток, А", self.secondary_current_combobox)
        tt_layout.addRow("Номинальная вторичная нагрузка, В*А", self.secondary_load_edit)
        tt_layout.addRow("Коэффициент мощности", self.power_factor_edit)
        tt_layout.addRow("Номинальная предельная картность", self.max_k_factor_edit)
        tt_layout.addRow("Ток намагничивания, А", self.excitation_current_edit)
        tt_layout.addRow("Число витков вторичной обмотки", self.secondary_winding_turns_edit)
        tt_layout.addRow("Активное сопротивление вторичной обмотки, Ом", self.secondary_winding_active_resistance_edit)
        tt_layout.addRow("Реактивное сопротивление вторичной обмотки, Ом", self.secondary_winding_reactive_resistance_edit)
        tt_layout.addRow("Индуктивность вторичной обмотки, Гн", self.secondary_winding_inductance_edit)
        tt_layout.addRow("Коэффициент остаточной намагниченности", self.remanence_coef_edit)
        tt_layout.addRow("Cечение стали магнитопровода, м2", self.cross_magnet_steel_edit)
        tt_layout.addRow("Cредняя длина силовой линии магнитного поля, м", self.aver_leng_magn_field_line_edit)


        tt_group_box.setLayout(tt_layout)

        # Данные Сети
        network_group_box = QGroupBox("Данные сети")
        network_layout = QFormLayout()
        self.damping_time_edit = QLineEdit()
        self.fault_current_edit = QLineEdit()

        network_layout.addRow("Время затухания апериодической составляющей, с", self.damping_time_edit)
        network_layout.addRow("Значение тока КЗ, А", self.fault_current_edit)

        network_group_box.setLayout(network_layout)

        # Данные Нагрузки
        load_group_box = QGroupBox("Данные нагрузки")
        load_layout = QFormLayout()
        self.load_active_resistance_edit = QLineEdit()
        self.load_reactive_resistance_edit = QLineEdit()
        self.load_inductance_edit = QLineEdit()

        load_layout.addRow("Фактическое активное сопротивление нагрузки, Ом", self.load_active_resistance_edit)
        load_layout.addRow("Фактическое реактивное сопротивление нагрузки, Ом", self.load_reactive_resistance_edit)
        load_layout.addRow("Индуктивность нагрузки, Гн", self.load_inductance_edit)

        load_group_box.setLayout(load_layout)

        # Кнопки
        button_layout = QHBoxLayout()
        calculate_button = QPushButton("Рассчитать")
        save_button = QPushButton("Сохранить запись")
        load_button = QPushButton("Загрузить запись")

        button_layout.addWidget(calculate_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(load_button)

        form_layout.addRow(tt_group_box)
        form_layout.addRow(network_group_box)
        form_layout.addRow(load_group_box)
        form_layout.addRow(button_layout)

        self.input_tab.setLayout(form_layout)

        # Сигналы кнопок
        calculate_button.clicked.connect(self.calculate_data)
        save_button.clicked.connect(self.save_data)
        load_button.clicked.connect(self.load_data)

    def init_result_tab(self):
        layout = QVBoxLayout()

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        self.result_tab.setLayout(layout)

    def init_graph_tab(self):
        layout = QVBoxLayout()

        # Создание графического элемента FigureCanvas
        self.canvas = FigureCanvas(Figure(figsize=(5, 4), dpi=100))
        layout.addWidget(self.canvas)

        self.graph_tab.setLayout(layout)
    
    def init_graph_anm_tab(self):
        layout = QVBoxLayout()

        # Создание графического элемента FigureCanvas
        self.canvas_2 = FigureCanvas(Figure(figsize=(5, 4), dpi=100))
        layout.addWidget(self.canvas_2)

        self.graph_tab.setLayout(layout)


    def calculate_data(self):
        voltage = float(self.voltage_edit.text())
        frequency = float(self.frequency_combobox.currentText())
        primary_current = float(self.primary_current_edit.text())
        secondary_current = float(self.secondary_current_combobox.currentText())
        secondary_load = float(self.secondary_load_edit.text())
        power_factor = float(self.power_factor_edit.text())
        max_k_factor = float(self.max_k_factor_edit.text())
        excitation_current = float(self.excitation_current_edit.text())
        secondary_winding_turns = float(self.secondary_winding_turns_edit.text())
        secondary_winding_active_resistance = float(self.secondary_winding_active_resistance_edit.text())
        secondary_winding_reactive_resistance = float(self.secondary_winding_reactive_resistance_edit.text())
        secondary_winding_inductance = float(self.secondary_winding_inductance_edit.text())
        damping_time = float(self.damping_time_edit.text())
        fault_current = float(self.fault_current_edit.text())
        load_active_resistance = float(self.load_active_resistance_edit.text())
        load_reactive_resistance = float(self.load_reactive_resistance_edit.text())
        load_inductance = float(self.load_inductance_edit.text())
        remanence_coef = float(self.remanence_coef_edit.text())
        cross_magnet_steel = float(self.cross_magnet_steel_edit.text())
        aver_leng_magn_field_line = float(self.aver_leng_magn_field_line_edit.text())

        power_factor_angle = math.acos(power_factor)

        angle_frequency = 2 * math.pi * frequency

        z_full_load = math.sqrt(load_active_resistance**2 + load_reactive_resistance**2)

        z2_full_nom = math.sqrt(
            (secondary_winding_active_resistance + z_full_load * power_factor)**2 + (secondary_winding_reactive_resistance + z_full_load * math.sin(power_factor_angle))**2
        )

        z2_full = math.sqrt(
            (secondary_winding_active_resistance + load_active_resistance)**2 + (secondary_winding_reactive_resistance + load_reactive_resistance)**2
        )

        parametr_a = (
            (primary_current * max_k_factor * z2_full_nom) /
              (fault_current * z2_full)
        )

        saturation_time = damping_time * math.log(
            (angle_frequency * damping_time) / 
              (angle_frequency * damping_time - (parametr_a * ( 1 - remanence_coef)) + 1)
        ) 


        cos_alpha = (
            (secondary_winding_active_resistance + load_active_resistance) / (z2_full)
        )

        Kpr_tnas = parametr_a * ( 1 - remanence_coef )

        alpha = math.acos(cos_alpha)

        r2 = secondary_winding_active_resistance * (z_full_load**2) / z2_full_nom**2
        x2 = secondary_winding_reactive_resistance * (z_full_load**2) / z2_full_nom**2

        i2_full_nom = secondary_current * voltage / (math.sqrt(3) * secondary_winding_turns)

        u2_full_nom = voltage / math.sqrt(3)

        i2_full = u2_full_nom / z2_full

        i2 = secondary_load / z2_full

        r1 = voltage / primary_current

        z1 = voltage / primary_current

        r2_z1 = r2 / z1

        t1 = r1 * z1

        t2 = (r2 * z1) / (z2_full_nom**2)

        t3 = (r2 * z1) / (z2_full_nom * fault_current)

        t4 = (r2_z1 * r2) / fault_current

        t5 = (r2_z1 * fault_current) / z2_full_nom

        t6 = (t1 - t2) / (t1 - t3)

        if t4**2 - t5**2 >= 0:
            t7 = math.sqrt(t4**2 - t5**2)
        else:
            t7 = 0

        t8 = t6 - t7

        t9 = (t1 - t8) / (t1 - t3)

        t10 = (t9 * z2_full_nom * fault_current) / r2

        t11 = r2 * fault_current / (z2_full_nom * t9)

        t12 = (t10 * r2) / (r2 + t11)

        t13 = (t12 * z2_full_nom * t9) / r2

        t14 = (t13 - t10) / (t13 - t11)

        if t4**2 - t5**2 * t14**2 >= 0:
            t15 = math.sqrt(t4**2 - t5**2 * t14**2)
        else:
            t15 = 0 

        i2_short_circuit = (t13 * fault_current) / (t13 - t11)

        u2_short_circuit = voltage / (t9 * (1 + t14) - t10 * t14)

        i2_short_circuit_norm = i2_short_circuit / i2_full_nom

        u2_short_circuit_norm = u2_short_circuit / u2_full_nom

        max_k_factor_nom = u2_short_circuit_norm / i2_short_circuit_norm

        max_k_factor_pu = max_k_factor / max_k_factor_nom

        u1_full_load = primary_current * z1

        u1_short_circuit = fault_current * z1

        u2_nom = math.sqrt(3) * u2_full_nom

        u1_short_circuit_pu = u1_short_circuit / u1_full_load

        u2_short_circuit_pu = u2_short_circuit / u2_full_nom

        max_k_factor_actual = u2_short_circuit_pu / i2_short_circuit_norm



        # Создание итогового отчета
        report = f"Итоговый отчет:\n\n"

        report += f"Значение полного сопротивления вторичной обмотки при номинальном токе, Z2фпн = {round(z2_full_nom,4) } Ом\n"
        report += f"Активное сопротивление вторичной обмотки, R2 = {round(r2,4) } Ом\n"
        report += f"Реактивное сопротивление вторичной обмотки, X2 = {round(x2,4) } Ом\n"
        report += f"Номинальный ток вторичной обмотки, I2фн = {round(i2_full_nom,4) } А\n"
        report += f"Номинальное напряжение на вторичной обмотке, U2фн = {round(u2_full_nom,4) } кВ\n"
        report += f"Ток вторичной обмотки, I2ф = {round(i2_full,4) } А\n"
        report += f"Эффективное значение тока вторичной обмотки, I2 = {round(i2,4) } А\n"
        report += f"Активное сопротивление первичной обмотки, R1 = {round(z1,4) } Ом\n"
        report += f"Сопротивление первичной обмотки, Z1 = {round(z1,4) } Ом\n"
        report += f"Отношение активного сопротивления вторичной обмотки к сопротивлению первичной обмотки, R2 / Z1 = {round(r2_z1,4) } о.е\n"
        report += f"Ток короткого замыкания на вторичной обмотке, I2КЗ = {round(i2_short_circuit,4) } А\n"
        report += f"Напряжение короткого замыкания на вторичной обмотке, U2КЗ = {round(u2_short_circuit,4) } кВ\n"
        report += f"Ток короткого замыкания на вторичной обмотке в относительных единицах к номинальному току, I2КЗ/фн = {round(i2_short_circuit_norm,4) } А\n"
        report += f"Напряжение короткого замыкания на вторичной обмотке в относительных единицах к номинальному напряжению, U2КЗ/фн = {round(u2_short_circuit_norm,4) } о.е\n"
        report += f"Максимальное значение номинальной картности: {round(max_k_factor_nom,4) }\n"
        report += f"Значение номинальной картности: {round(max_k_factor,4) }\n"
        report += f"Максимальное значение относительной картности: {round(max_k_factor_actual,4) }\n"
        report += f"Значение относительной картности: {round(max_k_factor_pu,4) }\n"
        report += f"Напряжение на первичной обмотке при нагрузке, U1Нагр = {round(u1_full_load,4) } кВ\n"
        report += f"Напряжение на первичной обмотке при коротком замыкании, U1КЗ = {round(u1_short_circuit,4) } кВ\n"
        report += f"Номинальное напряжение на вторичной обмотке, U2Ном = {round(u2_nom,4) } кВ\n"
        report += f"Напряжение на первичной обмотке в относительных единицах к номинальному напряжению, U1КЗ/ф = {round(u1_short_circuit_pu,4) } о.е\n"
        report += f"Напряжение на вторичной обмотке в относительных единицах к номинальному напряжению, U2КЗ/фн = {round(u2_short_circuit_pu,4) } о.е\n"

        self.result_text.setText(report)

        # Отобразить графики
        self.plot_graphs(primary_current, secondary_winding_turns, damping_time, secondary_current)



    def plot_graphs(self, primary_current, secondary_winding_turns, damping_time, secondary_current):    
        def equations(x, t):
            a = 10**(-12)
            b = 19.04
            c = 18.1

            B, i2 = x

            w1 = 1
            time = sympy.Symbol('time')
            
            i1_period = sympy.sqrt(2) * primary_current * sympy.sin(2 * sympy.pi * frequency * time)
            i1_aperiod = sympy.sqrt(2) * primary_current * sympy.exp( - time / damping_time)
            i1 = i1_period + i1_aperiod

            w2 = float(self.secondary_winding_turns_edit.text())
            R2 = float(self.secondary_winding_active_resistance_edit.text())
            L2 = float(self.secondary_winding_inductance_edit.text())
            Rn = float(self.load_active_resistance_edit.text())
            Ln = float(self.load_inductance_edit.text())
            s = float(self.cross_magnet_steel_edit.text())
            lcr = float(self.aver_leng_magn_field_line_edit.text())
            i1_diff = sympy.diff(i1, time)
            i1_diff_value = i1_diff.subs(time, t)
            print(t)

            F1 = (w2 * i2 * (R2 + Rn) + (L2 + Ln) * w1 * i1_diff_value)/(w2 * s * w2 + (L2 + Ln) * lcr * (a * b * mpmath.cosh(b * B) + c))
            F2 = (-lcr * (a * b * mpmath.cosh(b * B) + c) * i2 * (R2 + Rn) + w2 * s * w1 * i1_diff_value)/(w2 * s * w2 + (L2 + Ln) * lcr * (a * b * mpmath.cosh(b * B) + c))
            return [F1, F2]
        
        # Создать новый график
        fig = self.canvas.figure
        fig_2 = self.canvas_2.figure

        # Очистить предыдущие графики, если они есть
        fig.clear()
        fig_2.clear()

        # Создать оси графика
        ax = fig.add_subplot(111)
        ax_2 = fig_2.add_subplot(111)

        t_start = 0  # начальное время
        t_end = 10 # конечное время

        time = np.linspace(t_start, t_end, 200)

        #Работа с первичным током

        B0 = 0
        i20 = 0

        frequency = float(self.frequency_combobox.currentText())
        
        # i1_graph = primary_current * math.sqrt(2) * (1 + np.exp(-time/damping_time)) * np.sin(2 * np.pi * frequency * time)

        i1_graph_period = math.sqrt(2) * primary_current * np.sin(2 * np.pi * frequency * time)
        i1_graph_aperiod = math.sqrt(2) * primary_current * np.exp( - time / damping_time)
        i1_graph = i1_graph_aperiod + i1_graph_period
        y = [B0, i20]
        t = [t_start, t_end]

        solution = scipy.integrate.odeint(equations, y, time)
        solution_at_time = solution.T
        B = solution_at_time[0]
        i2_a = solution_at_time[1]

        #Построить график
        ax.plot(time, i1_graph, label='Первичный ток')
        ax.plot(time, i2_a * secondary_winding_turns, label='Вторичный ток')
    
        #Настроить оси и легенду
        ax.set_xlabel('Время, мс')
        ax.set_ylabel('Ток, А')
        ax.grid(True)
        ax.legend()

        B_2 = np.linspace(-2, 2, 100)
        H_2 = 10**-12 * np.sinh(19.04 * B_2) + 18.1 * B_2
        ax_2.plot(H_2, B_2, label='Кривая намагничивания')
        ax_2.set_ylabel('Магнитная индукция (B) в Тл')
        ax_2.set_xlabel('Магнитная индукция (H) в А/м')
        ax_2.grid(True)
        ax_2.legend()

        # Обновить графический элемент FigureCanvas
        self.canvas.draw()
        self.canvas_2.draw()



    def save_data(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, "Сохранить запись", "", "JSON (*.json)")

        if file_path:
            data = {
                "voltage": self.voltage_edit.text(),
                "frequency": self.frequency_combobox.currentText(),
                "primary_current": self.primary_current_edit.text(),
                "secondary_current": self.secondary_current_combobox.currentText(),
                "secondary_load": self.secondary_load_edit.text(),
                "power_factor": self.power_factor_edit.text(),
                "max_k_factor": self.max_k_factor_edit.text(),
                "excitation_current": self.excitation_current_edit.text(),
                "secondary_winding_turns": self.secondary_winding_turns_edit.text(),
                "secondary_winding_active_resistance": self.secondary_winding_active_resistance_edit.text(),
                "secondary_winding_reactive_resistance": self.secondary_winding_reactive_resistance_edit.text(),
                "secondary_winding_inductance": self.secondary_winding_inductance_edit.text(),
                "damping_time": self.damping_time_edit.text(),
                "fault_current": self.fault_current_edit.text(),
                "load_active_resistance": self.load_active_resistance_edit.text(),
                "load_reactive_resistance": self.load_reactive_resistance_edit.text(),
                "load_inductance": self.load_inductance_edit.text(),
                "remanence_coef": self.remanence_coef_edit.text(),
                "cross_magnet_steel": self.cross_magnet_steel_edit.text(),
                "aver_leng_magn_field_line": self.aver_leng_magn_field_line_edit.text()
            }

            with open(file_path, "w") as file:
                json.dump(data, file)
    
    def load_data(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Загрузить запись", "", "JSON (*.json)")

        if filename:
            with open(filename, "r") as file:
                data = json.load(file)

            self.voltage_edit.setText(data["voltage"])
            self.frequency_combobox.setCurrentText(data["frequency"])
            self.primary_current_edit.setText(data["primary_current"])
            self.secondary_current_combobox.setCurrentText(data["secondary_current"])
            self.secondary_load_edit.setText(data["secondary_load"])
            self.power_factor_edit.setText(data["power_factor"])
            self.max_k_factor_edit.setText(data["max_k_factor"])
            self.excitation_current_edit.setText(data["excitation_current"])
            self.secondary_winding_turns_edit.setText(data["secondary_winding_turns"])
            self.secondary_winding_active_resistance_edit.setText(data["secondary_winding_active_resistance"])
            self.secondary_winding_reactive_resistance_edit.setText(data["secondary_winding_reactive_resistance"])
            self.secondary_winding_inductance_edit.setText(data["secondary_winding_inductance"])
            self.damping_time_edit.setText(data["damping_time"])
            self.fault_current_edit.setText(data["fault_current"])
            self.load_active_resistance_edit.setText(data["load_active_resistance"])
            self.load_reactive_resistance_edit.setText(data["load_reactive_resistance"])
            self.load_inductance_edit.setText(data["load_inductance"])
            self.remanence_coef_edit.setText(data["remanence_coef"])
            self.cross_magnet_steel_edit.setText(data["cross_magnet_steel"])
            self.aver_leng_magn_field_line_edit.setText(data["aver_leng_magn_field_line"])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TransformerAnalysisApp()
    window.show()
    sys.exit(app.exec_())
