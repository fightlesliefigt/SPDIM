import torch
import random
import math
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
from sklearn.model_selection import KFold
import numpy as np

from typing import Iterator

class DomainDataset(Dataset):

    def __init__(self,
                 features: torch.Tensor,
                 labels: torch.LongTensor,
                 domains: torch.LongTensor,
                 label_ratio: dict,
                 seed_value=0):
        self.features = features
        self.domains = domains
        self.labels = labels
        self.imbalance_ratio = label_ratio
        self.seed_value = seed_value
        self.new_features = torch.Tensor()
        self.new_domains = torch.LongTensor()
        self.new_labels = torch.LongTensor()
        self.get_modified_data(self.domains, self.labels, self.imbalance_ratio)
        self.features = self.new_features
        self.domains = self.new_domains
        self.labels = self.new_labels

    def get_modified_data(self, domains, labels, imbalance_ratio):
        
        doms, domidx = domains.unique(return_inverse=True)
        doms = doms.tolist()
        domidx = domidx.tolist()
        self.indices = {}


        for i, domain in enumerate(doms):
             self.indices[domain] = [index for index, value in enumerate(domidx) if value == i] 
             clas,  classidx = labels[self.indices[domain]].unique(return_inverse=True)    
             clas = clas.tolist()
             classidx = classidx.tolist()  


             for j , clss in enumerate(clas):
                 modify_classidx = [self.indices[domain][idx] for idx, value in enumerate(classidx) if value == j ]
                 self.modify_data_for_domain_class(modify_classidx, imbalance_ratio[(clss)])

    def modify_data_for_domain_class(self, modify_classidx, imbalance_ratio):

        num_samples_to_keep = math.ceil(len(modify_classidx) * (imbalance_ratio))
        random.seed(self.seed_value)
        class_indices = random.sample(modify_classidx, num_samples_to_keep)
        self.new_features = torch.cat((self.features[class_indices], self.new_features))
        self.new_domains = torch.cat((self.domains[class_indices], self.new_domains))
        self.new_labels = torch.cat((self.labels[class_indices], self.new_labels))

    def update_labels(self, new_labels):
        self.labels = new_labels

    def __len__(self):
        return self.new_features.shape[0]

    def __getitem__(self, index):
        return [dict(inputs=self.features[index], domains=self.domains[index]), self.labels[index]]


class StratifiedDomainDataLoader(DataLoader):

    def __init__(self, dataset=None, batch_size=1, domains_per_batch=1, shuffle=True, **kwargs):

        if isinstance(dataset, Subset) and isinstance(dataset.dataset, Subset) and isinstance(dataset.dataset.dataset, DomainDataset):
            domains = dataset.dataset.dataset.domains[dataset.dataset.indices][dataset.indices]
            labels = dataset.dataset.dataset.labels[dataset.dataset.indices][dataset.indices]
        elif isinstance(dataset, Subset) and isinstance(dataset.dataset, DomainDataset):
            domains = dataset.dataset.domains[dataset.indices]
            labels = dataset.dataset.labels[dataset.indices]
        elif isinstance(dataset, DomainDataset):
            domains = dataset.domains
            labels = dataset.labels
        else:
            raise NotImplementedError()

        sampler = StratifiedDomainSampler(domains, labels,int(batch_size / domains_per_batch), domains_per_batch, shuffle=shuffle)

        super().__init__(dataset=dataset, sampler=sampler, batch_size=batch_size, **kwargs)


class StratifiedDomainSampler():

    def __init__(self, domains, stratvar, samples_per_domain, domains_per_batch, shuffle=True) -> None:
        self.samples_per_domain = samples_per_domain
        self.domains_per_batch = domains_per_batch
        self.shuffle = shuffle
        self.stratvar = stratvar

        du, didxs = domains.unique(return_inverse=True)
        du = du.tolist()
        didxs = didxs.tolist()
        
        self.domain_num = len(du)
        self.domaindict = {}
        self.domain_class_dict = {}
        self.domain_class_counts = {}
        self.domain_class_num = {}
        for domain, _ in enumerate(du):
            self.domaindict[domain] = torch.LongTensor([idx for idx, dom in enumerate(didxs) if dom == domain])   
            clas,  classidx = stratvar[self.domaindict[domain]].unique(return_inverse=True)
            clas = clas.tolist()
            classidx = classidx.tolist()     
            for j , classs in enumerate(clas):
                self.domain_class_dict[(domain, classs)] = [self.domaindict[domain][idx] for idx, value in enumerate(classidx) if value == j ] 
                self.domain_class_counts[(domain, classs)] = len(self.domain_class_dict[(domain, classs)]) 
        
        n_class = len(clas)

        if len(du) < self.domains_per_batch:        
            self.domains_per_batch = len(du)              
            self.samples_per_domain = int(samples_per_domain * domains_per_batch / self.domains_per_batch)
            self.class_sample_per_batch = int(self.samples_per_domain / n_class)
            self.samples_per_domain = self.class_sample_per_batch * n_class
           
        else:
            self.class_sample_per_batch = int(self.samples_per_domain / n_class)
            self.samples_per_domain = self.class_sample_per_batch * n_class
    
        for key in self.domain_class_counts:
            self.domain_class_num[key] = math.ceil(self.domain_class_counts[key] / self.class_sample_per_batch) 

        self.max_value = max(self.domain_class_num.values())
        self.min_value = min(self.domain_class_num.values())
        self.batch_num = self.max_value

        


    def __iter__(self) -> Iterator[int]:

        domaincounts = [self.batch_num] * self.domain_num 

        generators = {}
        for i in self.domain_class_dict.keys():
            if self.shuffle:
                permidxs = torch.randperm(len(self.domain_class_dict[i]))
            else:
                permidxs = range(len(self.domain_class_dict[i]))
            generators[i] = \
                iter(
                     KSampler(
                        self.stratvar[self.domain_class_dict[i],],
                        batch_size=self.class_sample_per_batch,
                        sample_time= math.ceil(self.batch_num * self.class_sample_per_batch / self.domain_class_counts[i]),
                        shuffle=self.shuffle
                    ))

        while sum(domaincounts) > 0:

            candidates = [idx for idx, num in enumerate(domaincounts) if num != 0]
            if len(candidates) < self.domains_per_batch:
                break

            permidxs = np.random.permutation(len(candidates))
            candidates = [candidates[i] for i in permidxs]
            
            batchdomains = candidates[:self.domains_per_batch]
            

            batch = []
            for item in batchdomains:
                for key in self.domain_class_dict.keys():
                    if key[0] == item:
                        within_domain_idxs = [next(generators[key]) for _ in range(self.class_sample_per_batch)]
                        batch.extend(self.domain_class_dict[key][i] for i in within_domain_idxs)
                domaincounts[item] = domaincounts[item] - 1
                yield from batch
                batch.clear()

        yield from []

    def __len__(self) -> int:
        return  self.batch_num * self.samples_per_domain * self.domain_num

    
class KSampler(Sampler[int]):
    
    def __init__(self, stratvar, batch_size, sample_time, shuffle = True):
        self.n_splits = max(int(stratvar.shape[0] / batch_size), 2)
        self.sample_time = sample_time
        self.stratvar = stratvar
        self.shuffle = shuffle

    def gen_sample_array(self):
        if self.shuffle:
            random_states = [torch.randint(0,int(1e8),size=()).item() for _ in range(self.sample_time)]
        else:
            random_states = [None] * self.sample_time
        
        splits = [KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=random_state) for random_state in random_states]  
        
        indices = [test for s in splits for _, test in s.split(self.stratvar, self.stratvar)]
        
        return list(np.hstack(indices))

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.stratvar)