import apiClient from './api';

export const intelligenceService = {
  /**
   * Fetch complete URL & Host Intelligence
   * @param {string} url - Target URL or Domain
   */
  async getUrlIntelligence(url) {
    if (!url) throw new Error('A target URL or domain is required.');

    const candidateEndpoints = [
      `/api/intelligence/url`,
      `/api/url-intelligence`,
      `/api/host-intelligence`,
      `/intelligence`
    ];

    for (const endpoint of candidateEndpoints) {
      try {
        const response = await apiClient.post(endpoint, { url });
        return normalizeIntelligenceResponse(response.data, url);
      } catch (err) {
        if (err.status === 404) continue;
        throw err;
      }
    }

    throw new Error(
      'URL Intelligence endpoint not found on backend. Ensure /api/intelligence/url is implemented.'
    );
  }
};

function normalizeIntelligenceResponse(data, originalUrl) {
  if (!data) return null;

  return {
    url: data.url || originalUrl,
    domain: data.domain || extractDomain(originalUrl),
    domainAge: data.domain_age_days
      ? `${data.domain_age_days} days`
      : data.domain_age || 'Not Available',
    whois: {
      registrar: data.whois?.registrar || data.registrar || 'N/A',
      creationDate: data.whois?.created || data.created_at || 'N/A',
      expirationDate: data.whois?.expires || data.expiration_date || 'N/A',
      registrantCountry: data.whois?.country || data.country || 'N/A',
      nameServers: data.whois?.name_servers || data.dns_servers || []
    },
    network: {
      ip: data.ip || data.network?.ip || 'Resolution pending',
      asn: data.asn || data.network?.asn || 'N/A',
      asName: data.as_name || data.network?.as_name || 'N/A',
      prefix: data.prefix || data.network?.prefix || 'N/A',
      geo: {
        country: data.geo?.country || data.country || 'Unknown',
        city: data.geo?.city || data.city || 'Unknown',
        coordinates: data.geo?.loc || null
      }
    },
    dnsRecords: data.dns_records || data.dns || [],
    redirectChain: data.redirects || data.redirect_chain || [],
    extractedFeatures: data.features || {
      entropy: data.entropy || null,
      length: originalUrl.length,
      protocol: originalUrl.split(':')[0] || 'http',
      subdomains: extractSubdomains(originalUrl),
      hasIpInHost: /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/.test(originalUrl)
    },
    securityIndicators: data.indicators || data.threat_indicators || [],
    investigationNotes: data.notes || data.summary || null
  };
}

function extractDomain(url) {
  try {
    const parsed = new URL(url.startsWith('http') ? url : `http://${url}`);
    return parsed.hostname;
  } catch {
    return url.split('/')[0];
  }
}

function extractSubdomains(url) {
  try {
    const domain = extractDomain(url);
    const parts = domain.split('.');
    return parts.length > 2 ? parts.slice(0, parts.length - 2).join('.') : 'none';
  } catch {
    return 'none';
  }
}

export default intelligenceService;
