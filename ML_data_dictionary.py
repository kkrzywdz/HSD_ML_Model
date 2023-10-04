# ML_dictionary.py

"""
Dictionary Data Module

This module contains a lists with data used in the application.
"""

# No imports needed in this file

__version__ = '0.8.0'

tag_list = ['[adq]',
            '[bd_vran]',
            '[bkc]',
            '[cape blanco]',
            '[cert]',
            '[certs]',
            '[ci/cd]',
            '[cmdlet]',
            '[cpt]',
            '[crypto]',
            '[ddp]',
            '[doc]',
            '[dpdk]',
            '[eeupdate]',
            '[epct]',
            '[fxp]',
            '[haps]',
            '[ice]',
            '[ice_intree',
            '[ipdk]',
            '[iset]',
            '[lanconf]',
            '[lce]',
            '[mini pkg]',
            '[model]',
            '[nup]',
            '[nura]',
            '[nvmupdate]',
            '[ovs]',
            '[p4-sde]',
            '[p4c]',
            '[p4k8s]',
            '[p4sde]',
            '[package]',
            '[perf]',
            '[ptp]',
            '[performance]',
            '[preboot]',
            '[proset]',
            '[rdma]',
            '[readme]',
            '[sep]',
            '[simics]',
            '[skd]',
            '[switchdev]',
            '[uefi]',
            '[upstream]',
            '[vran]',
            '[xdp]']

other_list = ['gls ndsw'
              'certification',
              'adq',
              'bkc',
              'blk',
              'ci/cd',
              'compiler',
              'cpt',
              'dcb',
              'ddp',
              'doc',
              'dpdk',
              'eeupdate',
              'epct',
              'esxi',
              'ethernet cmdlet',
              'fxp',
              'gls',
              'geneve',
              'haps',
              'hyper-v',
              'iavf',
              'ice',
              'ipdk',
              'ipsec',
              'ipu update tool',
              'iset',
              'iwarp',
              'k8s',
              'kubernetes',
              'kvm',
              'lanconf',
              'latency',
              'lce',
              'legacy',
              'macsec',
              'mevts-ice',
              'net',
              'nup',
              'nura',
              'nvmupdate',
              'ovs',
              'p4c',
              'k8s',
              'p4k8s',
              'p4sde',
              'package',
              'performance',
              'pkg',
              'ptp',
              'powershell',
              'preboot',
              'proset',
              'qos',
              'quartzville',
              'rdma',
              'readme',
              'roce',
              'rping',
              'sdk',
              'sep',
              'simics',
              'siov',
              'sriov',
              'sr-iov',
              'switchdev',
              'uefi',
              'upstream',
              'vf',
              'vxlan',
              'vfdatapath',
              'virt',
              'virtio',
              'vm',
              'vmnic',
              'vmq',
              'vran',
              'xdp']

# keywords dictionary
# this is old list kept for reference


full_list = ['acc', 'access', 'action', 'adapters', 'adaptive', 'address', 'adq', 'affinity',
             'anvm', 'apf', 'api', 'arp', 'ate', 'autoneg', 'auxiliary', 'azure', 'balance',
             'balancing', 'bandwidth', 'bkc', 'bonding', 'bridge', 'broadcast', 'buffer',
             'buffers', 'byte', 'cables', 'callback', 'capabilities', 'capability', 'carrier',
             'certs', 'channel', 'checksum', 'cluster', 'cmdlet', 'configured', 'connectivity',
             'console', 'controller', 'converged', 'cores', 'counter', 'counters', 'cplane',
             'cpu', 'cpu', 'cq', 'crypto', 'ctrlr', 'cycle', 'dcb', 'dcbx', 'dcrlookup', 'ddk',
             'ddp', 'debug', 'devices', 'devlink', 'devmem', 'dhcp', 'diskless', 'dma', 'dmesg',
             'docker', 'documentation', 'dot1q', 'downgrade', 'downgrade', 'dpdk', 'dpdk', 'driver', 'dynamic', 'ecmp',
             'eepid', 'eeprom', 'eeupdate', 'encap', 'ens', 'eswitch', 'esx', 'esxcli', 'esxi', 'ethertype', 'ethtool',
             'fdir', 'filter', 'filters', 'firmware', 'flags', 'flags', 'flat', 'flow', 'format',
             'forwarded', 'forwarding', 'frames', 'freebsd', 'frequency', 'fuse', 'fw', 'fwd', 'fxp',
             'gateway', 'generic', 'gui', 'hang', 'haps', 'hdrs', 'header', 'helper', 'huge',
             'i40e', 'iavf', 'ibio', 'ice', 'idpf', 'iecm', 'imc', 'img', 'indrv', 'infiniband',
             'initialization', 'interrupt', 'interrupts', 'interval', 'inventory', 'io', 'ioctl',
             'iommu', 'iov', 'ip', 'ipdk', 'iperf', 'iperf3', 'ipmi', 'ipsec', 'ipv4', 'ipv6',
             'irdma', 'irdma', 'irdma', 'irq', 'iscsi', 'iset', 'isolation', 'isst', 'iwarp',
             'ixgbe', 'ixgben', 'ixia', 'jumbo', 'k8s', 'kverb', 'kvm', 'label', 'lag', 'lan',
             'lanconf', 'latency', 'lce', 'learning', 'legacy', 'libvirt', 'link', 'lldp', 'load',
             'localhost', 'loop', 'loopback', 'lpm', 'lspci', 'mac', 'macvlan', 'mailbox', 'manager',
             'mapping', 'mask', 'master', 'mbox', 'memory', 'message', 'metadata', 'meter', 'minicom',
             'mirror', 'mirroring', 'modify', 'modprobe', 'msg', 'msix', 'mtu', 'multicast', 'mutex',
             'native', 'ncsi', 'negative', 'negotiation', 'neighbor', 'netconf', 'netdev', 'netlink',
             'netlist', 'netmask', 'netns', 'netperf', 'network', 'networking', 'new', 'nfs', 'nft',
             'npi', 'nsx', 'ntuple', 'numa', 'nvgre', 'nvm', 'nvme', 'nvmeof', 'nvmupdate', 'nvmupdate',
             'nvmupdate', 'nvram', 'ocp3', 'ofed', 'offload', 'offload', 'offloads', 'oid', 'orom', 'p4',
             'p4c', 'p4ctl', 'p4include', 'p4runtime', 'p4sde', 'p4sem', 'package', 'package', 'packet',
             'pagesize', 'pair', 'parser', 'path', 'pcap', 'pci', 'pcie', 'perf', 'performance', 'pf',
             'physical', 'pi', 'pid', 'ping', 'pipeline', 'pkg', 'pkgmv', 'pktlen', 'pkts', 'pldm',
             'pmd', 'pointer', 'policer', 'pool', 'port', 'ports', 'pps', 'preboot', 'private', 'process',
             'processor', 'promiscuous', 'properties', 'proset', 'protocol', 'proxy', 'ptp', 'ptp',
             'ptp4l', 'ptp4l', 'ptp4l', 'ptype', 'ptypes', 'pxe', 'qemu', 'qos', 'query', 'queues',
             'rbp', 'rdma', 'readme', 'reboot', 'recover', 'redirect', 'register', 'registers',
             'registry', 'reset', 'restart', 'ring', 'roce', 'rocev2', 'rocky', 'route', 'router',
             'routes', 'routing', 'rpc', 'rping', 'rping', 'rsp', 'rules', 'rxcsum', 'salem', 'sctp',
             'sde', 'sdk', 'seconds', 'security', 'sequence', 'service', 'sfp', 'share', 'shutdown',
             'silicon', 'simics', 'size', 'smp', 'socket', 'sonic', 'spdk', 'sriov', 'ssh', 'static',
             'statistics', 'storage', 'subnet', 'switch', 'switchd', 'switchdev', 'switchdev', 'synce4l',
             'syscall', 'tagged', 'tcp', 'tcp4', 'tcpdump', 'tcpip', 'tcpv6', 'threshold', 'throughput',
             'timer', 'timestamp', 'tools', 'trunk', 'trusted', 'tso', 'txcsum', 'type', 'uart', 'udp',
             'udpv4', 'uefi', 'unicast', 'updated', 'upgrade', 'upip', 'upstream', 'uverbs', 'vectors',
             'verbose', 'vf', 'vfio', 'vfs', 'virtchnl', 'virtio', 'virtqueue', 'virtual', 'virtualization',
             'vlan', 'vlan', 'vm', 'vm3', 'vmdq', 'vmeta', 'vmkernel', 'vmnic', 'vmnic', 'vmnic4',
             'vmnics', 'vmq', 'vms', 'vmswitch', 'vmware', 'vnet', 'vnic', 'vport', 'vports', 'vran',
             'vsan', 'vsi', 'vswitch', 'vxlan', 'wake', 'windows', 'xdp'
             ]


'''
old_keywords = ['gls',
            # Apps&Tools - PROSET & Ethernet CMDLets
            'ethernet cmdlet', '[cmdlet]',
            # VRAN
            '[bd_vran]', 'vran', '[vran]', '[cape blanco]',
            # PreBoot
            '[preboot]', 'preboot', '[uefi]', 'uefi', 'legacy',
            # QV - apps
            '[nvmupdate]', 'nvmupdate', '[epct]', 'epct', '[nup]', 'nup',
            # QV - core
            'lanconf', 'eeupdate', 'quartzville', 'ipu update tool', 'nura',
            '[lanconf]', '[eeupdate]', '[nura]',
            # Certification
            '[cert]', '[certs]', 'Certification',
            # IPDK
            '[ipdk]', 'ipdk', '[skd]', 'sdk',
            # System - BKC
            'bkc', 'ci/cd', 'doc', 'readme',
            '[bkc]', '[ci/cd]', '[doc]', '[readme]',
            # Apps&Tools - ISET
            '[iset]', 'iset',
            # Apps&Tools - PROSET & Ethernet CMDLets
            '[proset]', 'proset', 'powershell',
            # Performance
            '[performance]', '[perf]', 'performance', 'latency',
            # XDP & ADQ
            '[adq]', 'adq', '[xdp]', 'xdp',
            # FXP
            '[fxp]', 'fxp', '[package]', '[ddp]', 'package', 'ddp', '[mini pkg]', 'pkg',
            # Crypto - LCE
            '[lce]', 'lce', '[crypto]', 'macsec', 'ipsec',
            # Crypto - ICE
            '[ice]', '[ice_intree', 'ice', 'mevts-ice',
            # DPDK
            '[dpdk]', 'dpdk',
            # P4SDE & P4OVS
            '[p4c]', 'p4c', '[cpt]', 'cpt', '[p4sde]',
            # P4C & CPT
            '[p4-sde]', '[ovs]', 'p4sde', 'ovs', 'compiler',
            # P4K8s
            '[p4k8s]', 'k8s', 'p4k8s', 'kubernetes',
            # switchdev
            '[switchdev]', 'switchdev',
            # RDMA
            '[rdma]', 'rdma', 'iwarp', 'roce', 'rping',
            # SEP
            '[sep]', 'sep', 'virtio', 'blk', 'net',
            # SIMICS
            '[simics]', 'simics', '[model]', 'haps', '[haps]',
            # upstream
            '[upstream]', 'upstream',
            # bnic
            'esxi', 'qos', 'dcb',
            # virt
            'hyper-v', 'kvm', 'vmq', 'siov', 'sriov', 'iavf', 'vmnic',
            'vf', 'virt', 'vm'
            ]

tag_list = [keyword for keyword in keywords if '[' in keyword]
other_list = [keyword for keyword in keywords if '[' not in keyword]

# Sort the lists
tag_list.sort()
other_list.sort()

# Print the sorted lists
print("tag_list =", tag_list)
print("other_list =", other_list)

'''