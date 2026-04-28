import React, { useState, useEffect } from 'react';

function ProtectionReport() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      const res = await fetch('/api/protection/reports');
      const data = await res.json();
      setReports(data.reports || []);
    } catch (error) {
      console.error('Failed to fetch reports:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="card">Loading forensic reports...</div>;
  }

  return (
    <div className="card" style={{ maxHeight: '800px', overflowY: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.25rem' }}>Forensic Audit Log</h2>
        <span className="media-meta">{reports.length} Audits Completed</span>
      </div>

      {reports.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
          <p>No scans performed yet.</p>
        </div>
      ) : (
        reports.map((report) => (
          <div key={report._id} className="media-item fade-in" style={{ border: '1px solid rgba(255,255,255,0.05)' }}>
            <div className="media-info">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <span className="media-name">Audit ID: {report.mediaHash.slice(0, 8)}...</span>
                  <span className="media-meta">
                    {new Date(report.timestamp).toLocaleString()}
                  </span>
                </div>
                <span className={`badge ${report.verdict === 'Real' ? 'badge-real' : 'badge-fake'}`}>
                  {report.verdict}
                </span>
              </div>

              <div style={{ marginTop: '1rem', display: 'flex', gap: '2rem' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '0.25rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Confidence Score</span>
                    <span style={{ color: report.confidence > 70 ? 'var(--accent)' : '#fbbf24' }}>{report.confidence}%</span>
                  </div>
                  <div style={{ background: 'rgba(255,255,255,0.05)', height: '4px', borderRadius: '2px' }}>
                    <div style={{ 
                      width: `${report.confidence}%`, 
                      height: '100%', 
                      background: report.confidence > 70 ? 'var(--accent)' : '#fbbf24',
                      borderRadius: '2px'
                    }}></div>
                  </div>
                </div>

                <div style={{ width: '100px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Spread</div>
                  <div style={{ 
                    fontSize: '0.9rem', 
                    fontWeight: '700', 
                    color: report.spreadLevel === 'High' ? '#ef4444' : '#10b981' 
                  }}>
                    {report.spreadLevel}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))
      )}
    </div>
  );
}

export default ProtectionReport;
