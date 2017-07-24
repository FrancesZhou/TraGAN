import numpy as np
import os

class get_all_data():
	def __init__(self, prefix, outprefix, txtlist, length):
		self.prefix = prefix
		self.outprefix = outprefix
		self.txtlist = txtlist
		self.length = length

	def create_sequences(self):
		for txt in self.txtlist:
			filename = self.prefix+txt+'.txt'
			with open(filename, 'r') as f:
				lines = f.readlines()
			lines = lines[:10]
			data = [line.split(',') for line in lines]
			taxiID = [int(line[1]) for line in data]
			lon_lat = [[float(x) for x in line[2:4]] for line in data]
			date = [line[6] for line in data]

# sequences.npy - raw sequences
# sequences_bound.npy - sequences confined in specific box
# sequences_grid.npy - sequences represented as grid
# all_seq.npy - 
class get_all_data2():
	def __init__(self, prefix, outprefix, length):
		self.lonlen = 0
		self.latlen = 0
		self.prefix = prefix
		self.outprefix = outprefix
		self.length = length

	def process_all(self, lonmin, lonmax, latmin, latmax, gran):
		#self.create_sequences()
		self.create_sequences_bound(lonmin, lonmax, latmin, latmax, gran)
		self.create_sequences_grid(lonmin, lonmax, latmin, latmax, gran)
		self.create_grid_seq(lonmin, lonmax, latmin, latmax, gran)
		self.create_train_test_data()

	def get_train_test_data(self):
		pre_train = np.load(self.outprefix+'pre_train.npy')
		train = np.load(self.outprefix+'train.npy')
		test = np.load(self.outprefix+'test.npy')
		return pre_train, train, test

	def create_train_test_data(self):
		all_seq = np.load(self.outprefix+'all_seq.npy')
		# shuffle
		index = np.arange(len(all_seq))
		np.random.shuffle(index)
		all_seq = all_seq[index]
		n = len(all_seq)/5
		pre_train = all_seq[0:2*n]
		train = all_seq[2*n:4*n]
		test = all_seq[4*n:]
		np.save(self.outprefix+'pre_train.npy', pre_train)
		np.save(self.outprefix+'train.npy', train)
		np.save(self.outprefix+'test.npy', test)
		return pre_train, train, test

	def create_sequences(self):
		files = os.listdir(self.prefix)
		sequences = []
		for f in files:
			filename = self.prefix+f
			if os.path.getsize(filename):
				with open(filename, 'r') as fid:
					lines = fid.readlines()
				data = [line.split(',') for line in lines]
				taxiID = int(data[0][0])
				lon_lat = [[float(x) for x in line[2:4]] for line in data]
				lon_lat = np.asarray(lon_lat)
				sequences.append({'id':taxiID, 'sequence':lon_lat})
				
		# save raw sequences for all taxiID
		np.save(self.outprefix+'sequences.npy', sequences)
		

	def create_sequences_bound(self, lonmin, lonmax, latmin, latmax, gran):
		# for all area in Beijing, [115.5 ~ 117.5], [39.5 ~ 41], gran: 0.01 -> 200*150
		# for 5 circles in Beijing, [116.0 ~ 116.8], [39.6 ~ 40.2], gran: 0.005 -> 160*120
		sequences = np.load(self.outprefix+'sequences.npy')
		sequences = sequences.tolist()
		sequences_bound = []
		maxlonlat = []
		minlonlat = []
		#error_i = []
		for i in range(0,len(sequences)):
			# if i==598:
			# 	print i
			taxi = sequences[i]
			if taxi['id']==1285:
				print 1285
			lon = taxi['sequence'][:,0]
			lat = taxi['sequence'][:,1]
			indexZero = np.union1d(np.nonzero(lon==0)[0], np.nonzero(lat==0)[0])
			indexlonOut = np.union1d(np.nonzero(lon<=lonmin)[0], np.nonzero(lon>lonmax)[0])
			indexlatOut = np.union1d(np.nonzero(lat<=latmin)[0], np.nonzero(lat>latmax)[0])
			indexOut = np.union1d(indexZero, np.union1d(indexlonOut, indexlatOut))
			# delete
			seq = np.delete(taxi['sequence'], indexOut, 0)
			if seq.size:
				sequences_bound.append({'id': taxi['id'], 'sequence': seq})
				#sequences[i]['sequence'] = seq
				maxlonlat.append(np.amax(seq, axis=0))
				minlonlat.append(np.amin(seq, axis=0))
			else:
				print taxi['id']
				#error_i.append(i)
		# for i in error_i:
		# 	del sequences[i]
		np.save(self.outprefix+'sequences_bound.npy',sequences_bound)
		# [minlon, minlat] = np.amin(np.asarray(minlonlat), axis=0)
		# [maxlon, maxlat] = np.amax(np.asarray(maxlonlat), axis=0)
		# print([maxlon, maxlat])
		# print([minlon, minlat])
		# ===============================================
		# convert to grid sequence
		#grid_seq = self.create_sequences_grid(lonmin, lonmax, latmin, latmax, gran)

	def create_sequences_grid(self, lonmin, lonmax, latmin, latmax, gran):
		# 
		sequences_bound = np.load(self.outprefix+'sequences_bound.npy')
		# sequences_bound = sequences_bound.tolist()
		# error = (sequences_bound<0).nonzero()
		# print error.shape
		lonlen = int(round((lonmax - lonmin)/gran))
		latlen = int(round((latmax - latmin)/gran))
		self.lonlen = lonlen
		self.latlen = latlen
		sequences_grid = []
		for i in range(0,len(sequences_bound)):
			taxi = sequences_bound[i]
			lonlat = np.asarray(taxi['sequence'])
			gridseq = np.floor((lonlat - np.array([lonmin, latmin]))/gran)
			gridseq = np.minimum(gridseq, [lonlen-1, latlen-1])
			if (gridseq<0).sum():
				print 'error'
			sequences_grid.append({'id': taxi['id'], 'sequence_grid': gridseq})
		# save as sequences_grid.npy
		np.save(self.outprefix+'sequences_grid.npy', sequences_grid)

	def create_grid_seq(self, lonmin, lonmax, latmin, latmax, gran):
		lonlen = int(round((lonmax-lonmin)/gran))
		latlen = int(round((latmax-latmin)/gran))
		self.lonlen = lonlen
		self.latlen = latlen
		sequences_grid = np.load(self.outprefix+'sequences_grid.npy')
		sequences_grid = sequences_grid.tolist()
		all_seq_for_all_taxi = []
		all_seq = []
		for taxi in sequences_grid:
			taxiseq = np.int_(taxi['sequence_grid'])
			taxiseq_id = self.get_grid_id(taxiseq[:,0], taxiseq[:,1])
			taxiseq, seq = self.get_seq_of_length_for_taxi(taxiseq_id)
			all_seq_for_all_taxi.append({'id': taxi['id'], 'grid_seq': taxiseq})
			if len(seq):
				all_seq.append(seq)
		all_seq = np.concatenate(all_seq, 0)
		np.save(self.outprefix+'all_seq_for_all_taxi.npy', all_seq_for_all_taxi)
		np.save(self.outprefix+'all_seq.npy', all_seq)
		return all_seq


	def get_seq_of_length_for_taxi(self, taxiseq_id):
		# delete duplica
		indexes = np.unique(taxiseq_id, return_index=True)[1]
		taxiseq = [taxiseq_id[index] for index in sorted(indexes)]
		if len(taxiseq)>self.length-1:
			count = len(taxiseq) - self.length + 1
			all_seq_of_length = np.zeros((count, self.length), dtype=np.int)
			for i in range(0, count):
				all_seq_of_length[i] = taxiseq[i:i+self.length]
		else:
			all_seq_of_length = []
		
		return taxiseq, all_seq_of_length

	def get_grid_id(self, lon_x, lat_y):
		grid_id = lat_y*self.lonlen + lon_x
		return grid_id

	def get_lon_lat_xy(self, grid_id):
		lat = grid_id/self.lonlen
		lon = grid_id - lat*self.lonlen
		return [lon, lat]

	def get_grid_neighbor(self, grid_id):
		index = [-1, 0, 1]
		[x, y] = self.get_lon_lat_xy(grid_id)
		n_x = x + index
		n_y = y + index
		neighbor = []
		for x_i in n_x:
			for y_i in n_y:
				neighbor.append(self.get_grid_id(x_i, y_i))
		del neighbor[4]
		return neighbor
