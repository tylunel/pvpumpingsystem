# import matplotlib.pyplot as plt

# import pvpumpingsystem.pvgeneration as pvgen
# import pvpumpingsystem.mppt as mppt
# import pvpumpingsystem.pump as pp
# import pvpumpingsystem.pipenetwork as pn
# import pvpumpingsystem.pvpumpsystem as pvps

# pvgen1 = pvgen.PVGeneration(
#     # Weather data path
#     weather_data_and_metadata=(
#         '../../pvpumpingsystem/data/weather_files/'
#         'TUN_Tunis.607150_IWEC.epw'),

#     # PV array parameters
#     pv_module_name='Canadian Solar CS5C 80M',  # Name of pv module to model
#     modules_per_string=4,
#     strings_in_parallel=1,

#     # Models used (check pvlib.modelchain for all available models)
#     orientation_strategy='south_at_latitude_tilt',  # or 'flat' or None
#     )

# mppt1 = mppt.MPPT(efficiency=0.96,
#                   idname='PCA-120-BLS-M2'
#                   )
# pump_sunpump = pp.Pump(path="../../pvpumpingsystem/data/pump_files/SCB_10_150_120_BL.txt")

# pipes1 = pn.PipeNetwork(h_stat=20,  # static head [m]
#                         l_tot=100,  # length of pipes [m]
#                         diam=0.05,  # diameter [m]
#                         material='plastic')

# pvps1 = pvps.PVPumpSystem(pvgen1,
#                           pump_sunpump,
#                           coupling='direct',  # to adapt: 'mppt' or 'direct',
#                           mppt=mppt1,
#                           pipes=pipes1)
# pvps1.run_model()   

# # effective irradiance on PV array
# plt.figure()
# plt.plot(pvps1.efficiency.index[24:47].hour,
#          pvps1.pvgeneration.modelchain.results.effective_irradiance[24:47])
# plt.xlabel('Hour')
# plt.title('Effective irradiance vs time')
# plt.xlim([-1,23])

# # PV electric power
# plt.figure()
# plt.plot(pvps1.efficiency.index[24:47].hour,
#          pvps1.efficiency.electric_power[24:47])
# plt.xlabel('Hour')
# plt.title('Electric power in vs time')

# # Water volume in reservoir and output flow rate
# plt.figure()
# plt.plot(pvps1.efficiency.index[24:47].hour,
#          pvps1.flow.Qlpm[24:47])
# plt.xlabel('Hour')
# plt.ylabel('Pump output flow-rate [L/min]')
# plt.title('Output flow-rate vs time')


# plt.show()
