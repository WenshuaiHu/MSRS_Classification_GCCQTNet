
def write2csv(args, PATH_result, num_result = 1):
    
    if args.dataset=='Houston':
        path = PATH_result + "{}".format(num_result) + '/accuracy_AFSLNN_houston_'+str(num_result)+'.csv'
    if args.dataset=='Trento':
        path = PATH_result + "{}".format(num_result) + '/accuracy_AFSLNN_trento_'+str(num_result)+'.csv'
    if args.dataset=='Berlin':
        path = PATH_result + "{}".format(num_result) +  '/accuracy_AFSLNN_berlin_'+str(num_result)+'.csv'    
    if args.dataset=='Augsburg':
        path = PATH_result + "{}".format(num_result) +  '/accuracy_AFSLNN_augsburg_'+str(num_result)+'.csv'
    if args.dataset=='pavia':
        path = PATH_result + "{}".format(num_result) +  '/accuracy_AFSLNN_pavia_'+str(num_result)+'.csv'
    if args.dataset=='salinas':
        path = PATH_result + "{}".format(num_result) +  '/accuracy_AFSLNN__salinas_'+str(num_result)+'.csv'
    if args.dataset=='Muufl':
        path = PATH_result + "{}".format(num_result) +  '/accuracy_AFSLNN_muufl_'+str(num_result)+'.csv'
		
    result_writer = open(path, 'w')
	
    #Houston
    if args.dataset=='Houston':    
        result_writer.write('step,OA,AA(1),AA(2),AA(3),AA(4),AA(5),AA(6),AA(7),AA(8),AA(9),AA(10),AA(11),AA(12),AA(13),AA(14),AA(15),All_AA,Kappa\n')
    # Trento
    if args.dataset=='Trento':    
        result_writer.write('step,OA,AA(1),AA(2),AA(3),AA(4),AA(5),AA(6),All_AA,Kappa\n')
    #Indian
    if args.dataset=='Berlin':
        result_writer.write('step,OA,AA(1),AA(2),AA(3),AA(4),AA(5),AA(6),AA(7),AA(8),All_AA,Kappa\n')
    #Ksc
    if args.dataset=='Augsburg':    
        result_writer.write('step,OA,AA(1),AA(2),AA(3),AA(4),AA(5),AA(6),AA(7),All_AA,Kappa\n')
    #Pavia
    if args.dataset=='pavia':    
        result_writer.write('step,OA,AA(1),AA(2),AA(3),AA(4),AA(5),AA(6),AA(7),AA(8),AA(9),All_AA,Kappa\n')
    #Salinas
    if args.dataset=='salinas':
        result_writer.write('step,OA,AA(1),AA(2),AA(3),AA(4),AA(5),AA(6),AA(7),AA(8),AA(9),AA(10),AA(11),AA(12),AA(13),AA(14),AA(15),AA(16),All_AA,Kappa\n')
                             
    if args.dataset=='Muufl':
        result_writer.write('step,OA,AA(1),AA(2),AA(3),AA(4),AA(5),AA(6),AA(7),AA(8),AA(9),AA(10),AA(11),All_AA,Kappa\n')  
    return result_writer
