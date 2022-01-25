from utils import ufl_poly_from_table_data
import numpy as np


materials = {}


# Copper
materials["Copper"] = \
    {"kappa": lambda T: ufl_poly_from_table_data(
        x=np.array(
            [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
             623.15, 673.15, 723.15, 773.15, 823.15, 873.15, 923.15,
             973.15, 1023.15, 1073.15, 1123.15, 1173.15, 1223.15,
             1273.15]),
        y=np.array(
            [401.0, 398.0, 395.0, 391.0, 388.0, 384.0, 381.0, 378.0,
             374.0, 371.0, 367.0, 364.0, 360.0, 357.0, 354.0, 350.0,
             347.0, 344.0, 340.0, 337.0, 334.0]),
        u=T, degree=1, num_pieces=1),
     "c": lambda T: ufl_poly_from_table_data(
         x=np.array(
             [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
              623.15, 673.15, 723.15, 773.15, 823.15, 873.15, 923.15,
              973.15, 1023.15, 1073.15, 1123.15, 1173.15, 1223.15,
              1273.15]),
         y=np.array(
             [388.0, 390.0, 394.0, 398.0, 401.0, 406.0, 410.0, 415.0,
              419.0, 424.0, 430.0, 435.0, 441.0, 447.0, 453.0, 459.0,
              466.0, 472.0, 479.0, 487.0, 494.0]),
         u=T, degree=2, num_pieces=1),
     "thermal_strain": (lambda T: ufl_poly_from_table_data(
         x=np.array(
             [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
              623.15, 673.15, 723.15, 773.15, 823.15, 873.15, 923.15,
              973.15, 1023.15, 1073.15, 1123.15, 1173.15]),
         y=np.array(
             [16.7, 17.0, 17.2, 17.5, 17.7, 17.8, 18.0, 18.1, 18.2,
              18.4, 18.5, 18.65, 18.8, 18.97, 19.14, 19.34, 19.55,
              19.78, 20.05]) * 1e-6,
         u=T, degree=3, num_pieces=1), 293.15),
     "rho": lambda T: ufl_poly_from_table_data(
         x=np.array(
             [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
              623.15, 673.15, 723.15, 773.15, 823.15, 873.15, 923.15,
              973.15, 1023.15, 1073.15, 1123.15, 1173.15]),
         y=np.array(
             [8940.0, 8926.0, 8903.0, 8879.0, 8854.0, 8829.0, 8802.0,
              8774.0, 8744.0, 8713.0, 8681.0, 8647.0, 8612.0, 8575.0,
              8536.0, 8495.0, 8453.0, 8409.0, 8363.0]),
         u=T, degree=2, num_pieces=1),
     "E": lambda T: ufl_poly_from_table_data(
         x=np.array(
             [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
              623.15, 673.15]),
         y=np.array(
             [117.0, 116.0, 114.0, 112.0, 110.0, 108.0, 105.0, 102.0,
              98.0]) * 1e9,
         u=T, degree=2, num_pieces=1),
     "nu": 0.33
     }

# Inconel 625
materials["Inconel 625"] = \
    {"kappa": lambda T: ufl_poly_from_table_data(
        x=np.array(
            [293.15, 373.15, 423.15, 473.15, 523.15, 573.15, 623.15, 673.15]),
        y=np.array(
            [9.8, 10.9, 11.7, 12.4, 13.2, 13.9, 14.7, 15.4]),
        u=T, degree=1, num_pieces=1),
     "c": lambda T: ufl_poly_from_table_data(
        x=np.array(
            [293.15, 373.15, 473.15, 573.15, 673.15]),
        y=np.array(
            [407.0, 419.0, 447.0, 475.0, 499.0]),
        u=T, degree=3, num_pieces=1),
     "E": lambda T: ufl_poly_from_table_data(
        x=np.array(
            [293.15, 373.15, 423.15, 473.15, 523.15, 573.15, 623.15, 673.15]),
        y=np.array(
            [207.0, 202.0, 199.0, 197.0, 194.0, 191.0, 189.0, 186.0]) * 1e9,
        u=T, degree=1, num_pieces=1),
     "thermal_strain": (lambda T: ufl_poly_from_table_data(
         x=np.array(
             [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
              623.15, 673.15, 723.15, 773.15, 823.15, 873.15, 923.15,
              973.15, 1023.15, 1073.15, 1123.15, 1173.15]),
         y=np.array(
             [16.7, 17.0, 17.2, 17.5, 17.7, 17.8, 18.0, 18.1, 18.2, 18.4,
              18.5, 18.65, 18.8, 18.97, 19.14, 19.34, 19.55, 19.78,
              20.05]) * 1e-6,
         u=T, degree=3, num_pieces=1), 293.15)}

materials["CuCrZr"] = \
    {"kappa": lambda T: ufl_poly_from_table_data(
        x=np.array(
            [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
             623.15, 673.15, 723.15, 773.15]),
        y=np.array(
            [318.0, 324.0, 333.0, 339.0, 343.0, 345.0, 346.0, 347.0,
             347.0, 346.0, 346.0]),
        u=T, degree=3, num_pieces=1),
     "c": lambda T: ufl_poly_from_table_data(
        x=np.array(
            [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
             623.15, 673.15, 723.15, 773.15, 823.15, 873.15, 923.15,
             973.15]),
        y=np.array(
            [20.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0,
             450.0, 500.0, 550.0, 600.0, 650.0, 700.0]),
        u=T, degree=1, num_pieces=1),
     "thermal_strain": (lambda T: ufl_poly_from_table_data(
         x=np.array(
             [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
              673.15, 723.15, 773.15, 823.15, 873.15]),
         y=np.array(
             [16.7, 17.0, 17.3, 17.5, 17.7, 17.8, 18.0, 18.1, 18.2, 18.4,
              18.5, 18.6]) * 1e-6,
         u=T, degree=3, num_pieces=1), 293.15),
     "rho": lambda T: ufl_poly_from_table_data(
        x=np.array(
            [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
             623.15, 673.15, 723.15, 773.15]),
        y=np.array(
            [8900.0, 8886.0, 8863.0, 8840.0, 8816.0, 8791.0, 8767.0,
             8742.0, 8716.0, 8691.0, 8665.0]),
        u=T, degree=2, num_pieces=1),
     "E": lambda T: ufl_poly_from_table_data(
        x=np.array(
            [293.15, 373.15, 423.15, 473.15, 523.15, 573.15, 623.15,
             673.15, 723.15, 773.15, 873.15, 973.15]),
        y=np.array(
            [127.5, 127.0, 125.0, 123.0, 121.0, 118.0, 116.0, 113.0,
             110.0, 106.0, 95.0, 86.0]) * 1e9,
        u=T, degree=2, num_pieces=1),
     }
