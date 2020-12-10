with open(r'C:\Users\chira\OneDrive\Documents\Uni\Thesis\VCS-MOSI-DEV-VINN\mosi_dev_vinn\ip1linelist.json') as json_file:
    ip1 = json.load(json_file)
	
ip1 = np.array(ip1) 
self.network.forward_pass(self.network, ip1, self.network.norm)