import apiClient from './api';

export const graphService = {
  /**
   * Fetch infrastructure graph data for a target URL or domain
   * @param {string} target - Target URL or domain
   */
  async getInfrastructureGraph(target) {
    if (!target) throw new Error('Target URL or Domain is required.');

    const candidateEndpoints = [
      `/api/graph/infrastructure`,
      `/api/infra-graph`,
      `/api/gnn/graph`,
      `/graph`
    ];

    for (const endpoint of candidateEndpoints) {
      try {
        const response = await apiClient.post(endpoint, { target });
        if (response.data && response.data.nodes && response.data.edges) {
          return {
            isLiveBackend: true,
            nodes: response.data.nodes,
            edges: response.data.edges,
            meta: response.data.meta || { gnnModel: 'URLInfraGNN Layer-3' }
          };
        }
      } catch (err) {
        if (err.status === 404) continue;
        throw err;
      }
    }

    throw new Error('Graph endpoint not found on backend.');
  },

  /**
   * Provides clearly labeled demonstration data representing URLInfraGNN graph topologies
   * Strictly flagged as DEMO mode so it is never misrepresented as live intelligence.
   */
  getDemoGraph(targetUrl = 'http://bank-secure-update.account-verification.xyz/login.php') {
    return {
      isLiveBackend: false,
      isDemoMode: true,
      label: '[DEMO TOPOLOGY] URLInfraGNN Graph Schema Simulation',
      nodes: [
        {
          id: 'url-root',
          type: 'urlNode',
          position: { x: 250, y: 50 },
          data: {
            label: targetUrl,
            category: 'URL (Target)',
            risk: 'Malicious',
            status: 'Target Ingestion',
            details: 'Observed phishing path imitating banking portal.'
          }
        },
        {
          id: 'domain-1',
          type: 'domainNode',
          position: { x: 100, y: 190 },
          data: {
            label: 'account-verification.xyz',
            category: 'Apex Domain',
            risk: 'Suspicious',
            registered: '7 days ago (Fast-Flux)',
            registrar: 'NameCheap Privacy Inc'
          }
        },
        {
          id: 'ip-1',
          type: 'ipNode',
          position: { x: 50, y: 340 },
          data: {
            label: '185.220.101.5',
            category: 'IP Address',
            risk: 'High Risk',
            country: 'RU',
            reverseDns: 'tor-exit.relay.net'
          }
        },
        {
          id: 'ip-2',
          type: 'ipNode',
          position: { x: 250, y: 340 },
          data: {
            label: '194.26.29.112',
            category: 'IP Address (C2 Fallback)',
            risk: 'Critical',
            country: 'NL',
            reverseDns: 'unallocated.nl'
          }
        },
        {
          id: 'asn-1',
          type: 'asnNode',
          position: { x: 150, y: 490 },
          data: {
            label: 'AS44050 (Bulletproof Hosting)',
            category: 'Autonomous System',
            risk: 'Known Malicious Bulletproof ISP',
            prefix: '185.220.100.0/22',
            asnName: 'Petersburg Telecom Group'
          }
        },
        {
          id: 'cert-1',
          type: 'certNode',
          position: { x: 420, y: 200 },
          data: {
            label: "Let's Encrypt Free TLS",
            category: 'SSL/TLS Certificate',
            risk: 'Disposable Certificate',
            issuer: 'R3 Authority',
            validDays: 90
          }
        }
      ],
      edges: [
        {
          id: 'e1',
          source: 'url-root',
          target: 'domain-1',
          label: 'RESOLVES_TO_DOMAIN',
          animated: true,
          style: { stroke: '#00f2fe', strokeWidth: 2 }
        },
        {
          id: 'e2',
          source: 'url-root',
          target: 'cert-1',
          label: 'USES_CERTIFICATE',
          style: { stroke: '#8b5cf6', strokeWidth: 1.5 }
        },
        {
          id: 'e3',
          source: 'domain-1',
          target: 'ip-1',
          label: 'A_RECORD',
          animated: true,
          style: { stroke: '#ef4444', strokeWidth: 2 }
        },
        {
          id: 'e4',
          source: 'domain-1',
          target: 'ip-2',
          label: 'CNAME_FALLBACK',
          style: { stroke: '#f59e0b', strokeWidth: 1.5, strokeDasharray: '4 4' }
        },
        {
          id: 'e5',
          source: 'ip-1',
          target: 'asn-1',
          label: 'ROUTED_THROUGH',
          style: { stroke: '#06b6d4', strokeWidth: 2 }
        },
        {
          id: 'e6',
          source: 'ip-2',
          target: 'asn-1',
          label: 'ROUTED_THROUGH',
          style: { stroke: '#06b6d4', strokeWidth: 2 }
        }
      ]
    };
  }
};

export default graphService;
