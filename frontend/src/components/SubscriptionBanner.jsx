/**
 * SubscriptionBanner — global warning banner for license status.
 *
 * Shows warnings for:
 * - Approaching expiry (< 7 days)
 * - Grace period (expired but within 7 days)
 * - Expired (read-only mode)
 * - Suspended (payment failed)
 * - User limit approaching (90%+)
 */

import { useState, useEffect } from 'react';
import { useLocale } from '../i18n';
import { getLicenseInfo } from '../api/license';
import { AlertTriangle, X, Clock, Users } from 'lucide-react';

export default function SubscriptionBanner() {
  const { t } = useLocale();
  const [banner, setBanner] = useState(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let mounted = true;

    const checkLicense = async () => {
      try {
        const { license } = await getLicenseInfo();
        if (!mounted) return;

        // Determine banner type
        if (license.status === 'expired') {
          setBanner({
            type: 'error',
            icon: AlertTriangle,
            message: t('subscription.expired'),
          });
        } else if (license.status === 'suspended') {
          setBanner({
            type: 'error',
            icon: AlertTriangle,
            message: t('subscription.suspended'),
          });
        } else if (license.status === 'grace_period') {
          setBanner({
            type: 'warning',
            icon: Clock,
            message: t('subscription.grace_period', { days: Math.abs(license.days_remaining || 0) }),
          });
        } else if (license.status === 'active' && license.days_remaining !== null && license.days_remaining <= 7) {
          setBanner({
            type: 'warning',
            icon: Clock,
            message: t('subscription.expiring_soon', { days: license.days_remaining }),
          });
        } else if (license.max_users > 1 && license.current_users / license.max_users >= 0.9) {
          setBanner({
            type: 'info',
            icon: Users,
            message: t('subscription.users_approaching', {
              current: license.current_users,
              max: license.max_users,
            }),
          });
        } else {
          setBanner(null);
        }
      } catch {
        // Silently fail — don't show banner if can't check
      }
    };

    checkLicense();
    const interval = setInterval(checkLicense, 5 * 60 * 1000); // Check every 5 minutes

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [t]);

  if (!banner || dismissed) return null;

  const colors = {
    error: 'bg-red-500/15 border-red-500/40 text-red-300',
    warning: 'bg-amber-500/15 border-amber-500/40 text-amber-300',
    info: 'bg-blue-500/15 border-blue-500/40 text-blue-300',
  };

  const Icon = banner.icon;

  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 border-b text-xs ${colors[banner.type]}`}>
      <Icon size={14} className="shrink-0" />
      <span className="flex-1">{banner.message}</span>
      <button
        onClick={() => setDismissed(true)}
        className="text-current opacity-50 hover:opacity-100 transition-opacity"
      >
        <X size={14} />
      </button>
    </div>
  );
}
