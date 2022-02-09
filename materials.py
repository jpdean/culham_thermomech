from utils import ufl_poly_from_table_data, ufl_linear_interp
import numpy as np


materials = {}

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
     "nu": 0.33
     }

materials["304SS"] = \
    {"kappa": lambda T: ufl_poly_from_table_data(
        x=np.array(
            [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
             623.15, 673.15, 723.15, 773.15, 823.15, 873.15, 923.15,
             973.15, 1023.15, 1073.15]),
        y=np.array(
            [14.28, 14.73, 15.48, 16.23, 16.98, 17.74, 18.49, 19.24,
             19.99, 20.74, 21.49, 22.24, 22.99, 23.74, 24.49, 25.25,
             26.0]),
        u=T, degree=1, num_pieces=1),
     "c": lambda T: ufl_poly_from_table_data(
        x=np.array(
            [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
             623.15, 673.15, 723.15, 773.15, 823.15, 873.15, 923.15,
             973.15, 1023.15, 1073.15]),
        y=np.array(
            [472.0, 485.0, 501.0, 512.0, 522.0, 530.0, 538.0, 546.0,
             556.0, 567.0, 578.0, 590.0, 601.0, 610.0, 615.0, 615.0,
             607.0]),
        u=T, degree=4, num_pieces=1),
     "thermal_strain": (lambda T: ufl_poly_from_table_data(
         x=np.array(
             [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
              623.15, 673.15, 723.15, 773.15, 823.15, 873.15, 923.15,
              973.15, 1023.15, 1073.15, 1123.15, 1173.15, 1223.15, 1273.15]),
         y=np.array(
             [15.3, 15.5, 15.9, 16.2, 16.6, 16.9, 17.2, 17.5, 17.8, 18.0,
              18.3, 18.5, 18.7, 18.9, 19.0, 19.2, 19.3, 19.5, 19.6, 19.7,
              19.7]) * 1e-6,
         u=T, degree=2, num_pieces=1), 293.15),
     "rho": lambda T: ufl_poly_from_table_data(
        x=np.array(
            [293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15,
             623.15, 673.15, 723.15, 773.15, 823.15, 873.15, 923.15,
             973.15, 1023.15, 1073.15]),
        y=np.array(
            [7930.0, 7919.0, 7899.0, 7879.0, 7858.0, 7837.0, 7815.0,
             7793.0, 7770.0, 7747.0, 7724.0, 7701.0, 7677.0, 7654.0,
             7630.0, 7606.0, 7582.0]),
        u=T, degree=2, num_pieces=1),
     "E": lambda T: ufl_poly_from_table_data(
        x=np.array(
            [293.15, 373.15, 423.15, 473.15, 523.15, 573.15, 623.15,
             673.15, 723.15, 773.15, 823.15, 873.15, 923.15, 973.15]),
        y=np.array(
            [200.0, 193.0, 189.0, 185.0, 180.0, 176.0, 172.0, 168.0,
             164.0, 159.0, 155.0, 151.0, 147.0, 142.0]) * 1e9,
        u=T, degree=1, num_pieces=1),
     "nu": 0.33}

materials["water"] = \
    {"h": lambda T: ufl_linear_interp(
        xs=np.array(
            [293.15, 294.15, 295.15, 296.15, 297.15, 298.15, 299.15, 300.15,
             301.15, 302.15, 303.15, 304.15, 305.15, 306.15, 307.15, 308.15,
             309.15, 310.15, 311.15, 312.15, 313.15, 314.15, 315.15, 316.15,
             317.15, 318.15, 319.15, 320.15, 321.15, 322.15, 323.15, 324.15,
             325.15, 326.15, 327.15, 328.15, 329.15, 330.15, 331.15, 332.15,
             333.15, 334.15, 335.15, 336.15, 337.15, 338.15, 339.15, 340.15,
             341.15, 342.15, 343.15, 344.15, 345.15, 346.15, 347.15, 348.15,
             349.15, 350.15, 351.15, 352.15, 353.15, 354.15, 355.15, 356.15,
             357.15, 358.15, 359.15, 360.15, 361.15, 362.15, 363.15, 364.15,
             365.15, 366.15, 367.15, 368.15, 369.15, 370.15, 371.15, 372.15,
             373.15, 374.15, 375.15, 376.15, 377.15, 378.15, 379.15, 380.15,
             381.15, 382.15, 383.15, 384.15, 385.15, 386.15, 387.15, 388.15,
             389.15, 390.15, 391.15, 392.15, 393.15, 394.15, 395.15, 396.15,
             397.15, 398.15, 399.15, 400.15, 401.15, 402.15, 403.15, 404.15,
             405.15, 406.15, 407.15, 408.15, 409.15, 410.15, 411.15, 412.15,
             413.15, 414.15, 415.15, 416.15, 417.15, 418.15, 419.15, 420.15,
             421.15, 422.15, 423.15, 424.15, 425.15, 426.15, 427.15, 428.15,
             429.15, 430.15, 431.15, 432.15, 433.15, 434.15, 435.15, 436.15,
             437.15, 438.15, 439.15, 440.15, 441.15, 442.15, 443.15, 444.15,
             445.15, 446.15, 447.15, 448.15, 449.15, 450.15, 451.15, 452.15,
             453.15, 454.15, 455.15, 456.15, 457.15, 458.15, 459.15, 460.15,
             461.15, 462.15, 463.15, 464.15, 465.15, 466.15, 467.15, 468.15,
             469.15, 470.15, 471.15, 472.15, 473.15, 474.15, 475.15, 476.15,
             477.15, 478.15, 479.15, 480.15, 481.15, 482.15, 483.15, 484.15,
             485.15, 486.15, 487.15, 488.15, 489.15, 490.15, 491.15, 492.15,
             493.15, 494.15, 495.15, 496.15, 497.15, 498.15, 499.15, 500.15,
             501.15, 502.15, 503.15, 504.15, 505.15, 506.15, 507.15, 508.15,
             509.15, 510.15, 511.15, 512.15]),
        ys=np.array(
            [2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.420000e+04, 2.420000e+04, 2.420000e+04,
             2.420000e+04, 2.451700e+04, 2.546500e+04, 2.703400e+04,
             2.920600e+04, 3.195700e+04, 3.525800e+04, 3.907200e+04,
             4.335700e+04, 4.806700e+04, 5.315000e+04, 5.855000e+04,
             6.420800e+04, 7.006200e+04, 7.604800e+04, 8.210000e+04,
             8.815200e+04, 9.413800e+04, 9.999200e+04, 1.056500e+05,
             1.110500e+05, 1.161300e+05, 1.208400e+05, 1.251300e+05,
             1.289400e+05, 1.322400e+05, 1.349900e+05, 1.371700e+05,
             1.387300e+05, 1.396800e+05, 1.400000e+05, 1.399700e+05,
             1.398600e+05, 1.396900e+05, 1.394500e+05, 1.391500e+05,
             1.387800e+05, 1.383400e+05, 1.378300e+05, 1.372600e+05,
             1.366200e+05, 1.359200e+05, 1.351600e+05, 1.343400e+05,
             1.334500e+05, 1.325100e+05, 1.315100e+05, 1.304500e+05,
             1.293400e+05, 1.281700e+05, 1.269500e+05, 1.256800e+05,
             1.243600e+05, 1.230000e+05, 1.215900e+05, 1.201300e+05,
             1.186400e+05, 1.171000e+05, 1.155300e+05, 1.139300e+05,
             1.122900e+05, 1.106200e+05, 1.089200e+05, 1.072000e+05,
             1.054500e+05, 1.036800e+05, 1.019000e+05, 1.000900e+05,
             9.827800e+04, 9.645000e+04, 9.461300e+04, 9.277100e+04,
             9.092400e+04, 8.907600e+04, 8.722900e+04, 8.538700e+04,
             8.355000e+04, 8.172200e+04, 7.990600e+04, 7.810300e+04,
             7.631700e+04, 7.454900e+04, 7.280300e+04, 7.107900e+04,
             6.938200e+04, 6.771300e+04, 6.607400e+04, 6.446800e+04,
             6.289700e+04, 6.136300e+04, 5.986800e+04, 5.841400e+04,
             5.700400e+04, 5.563800e+04, 5.431900e+04, 5.305000e+04,
             5.183000e+04, 5.066300e+04, 4.954900e+04, 4.849100e+04,
             4.748900e+04, 4.654600e+04, 4.566100e+04, 4.483800e+04,
             4.407600e+04, 4.337600e+04, 4.274100e+04, 4.217000e+04,
             4.166400e+04, 4.122400e+04, 4.085100e+04, 4.054500e+04,
             4.030700e+04, 4.013700e+04, 4.003400e+04, 4.000000e+04,
             4.003400e+04, 4.013700e+04, 4.030700e+04, 4.054500e+04,
             4.085100e+04, 4.122400e+04, 4.166400e+04, 4.217000e+04,
             4.274100e+04, 4.337600e+04, 4.407600e+04, 4.483800e+04,
             4.566100e+04, 4.654600e+04, 4.748900e+04, 4.849100e+04,
             4.954900e+04, 5.066300e+04, 5.183000e+04, 5.305000e+04,
             5.431900e+04, 5.563800e+04, 5.700400e+04, 5.841400e+04]),
        u=T)}
