/**
 * ClassificationPanel — domain classification management + create modal.
 * Extracted from AdminPage.jsx.
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocale } from '../../i18n';
import { listDomains, getDomainDetail, getActiveDomainConfig, setActiveDomain, saveDomainYaml } from '../../services/bridge';
import {
  Plus, ChevronDown, Copy, CheckCircle, AlertTriangle,
} from 'lucide-react';


// ── Prompt Generator ─────────────────────────────────────

function generateDomainPrompt({ domainId, nameEn, nameKo, description }) {
  return `You are a domain profile generator for ImageParser, an AI-powered image classification system.

Create a YAML domain profile for: "${nameEn}" (${nameKo})
Description/Use case: ${description}

## YAML Schema (MUST follow exactly)

domain:
  id: ${domainId}
  name: "${nameEn}"
  name_ko: "${nameKo}"
  description: "${description}"

image_types:
  - type1
  - type2
  # ... (at least 3, at most 12 types)

type_hints:
  type1:
    field_name:
      - value1
      - value2
    another_field:
      - value1
  type2:
    field_name:
      - value1
  # Each image_type MUST have at least one hint field
  # Each hint field MUST be a list of 3-15 string values
  # Values must be snake_case

type_instructions:
  type1: "Extra instruction text for AI when analyzing this type"
  # Optional per type, but recommended for at least 3 types

## Example (game_asset domain, abbreviated)

domain:
  id: game_asset
  name: "Game Asset"
  name_ko: "게임 에셋"
  description: "Game development assets including characters, backgrounds, UI elements, items, and effects."

image_types:
  - character
  - background
  - ui_element
  - item
  - icon
  - texture
  - effect

type_hints:
  character:
    character_type:
      - hero
      - villain
      - NPC
      - boss
      - monster
    pose:
      - standing
      - action
      - portrait
      - full_body
    emotion:
      - neutral
      - happy
      - angry
      - sad
      - determined
  background:
    scene_type:
      - dungeon
      - castle
      - forest
      - village
      - arena
    time_of_day:
      - dawn
      - morning
      - noon
      - sunset
      - night
  item:
    item_type:
      - weapon
      - armor
      - potion
      - accessory
      - material
    rarity_feel:
      - common
      - uncommon
      - rare
      - epic
      - legendary

type_instructions:
  character: "Identify character class/role from visual cues (armor type, weapons, accessories). Note if this is a sprite sheet or single illustration."
  background: "Pay special attention to game-specific scene elements like interactive objects, spawn points, and pathways."
  item: "Identify item rarity from visual cues (glow effects, color coding, particle effects)."

## Rules
1. Output ONLY the YAML content — no markdown fences, no explanations, no commentary
2. image_types: 3-12 items, lowercase snake_case
3. type_hints: every image_type MUST have at least 1 hint field with 3-15 values
4. Hint field names and values: snake_case
5. type_instructions: provide for at least the top 3 most complex types
6. Do NOT include common_hints (art_style, color_palette) — those come from _base.yaml automatically
7. domain.id MUST be exactly: ${domainId}
8. Make the hints genuinely useful for classifying "${nameEn}" images`;
}


// ── Create Domain Modal ──────────────────────────────────

function CreateDomainModal({ existingIds, onClose, onCreated }) {
  const { t } = useLocale();
  const [step, setStep] = useState(1);
  const [domainId, setDomainId] = useState('');
  const [nameEn, setNameEn] = useState('');
  const [nameKo, setNameKo] = useState('');
  const [description, setDescription] = useState('');
  const [generatedPrompt, setGeneratedPrompt] = useState('');
  const [yamlInput, setYamlInput] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  const idError = domainId && !/^[a-z][a-z0-9_]*$/.test(domainId)
    ? t('admin.classification_id_invalid')
    : domainId && existingIds.includes(domainId)
      ? t('admin.classification_id_exists')
      : '';

  const canNext1 = domainId && nameEn && nameKo && description && !idError;

  const handleNext1 = () => {
    const prompt = generateDomainPrompt({ domainId, nameEn, nameKo, description });
    setGeneratedPrompt(prompt);
    setStep(2);
  };

  const handleCopyPrompt = async () => {
    try {
      await navigator.clipboard.writeText(generatedPrompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for non-HTTPS contexts
      const textarea = document.createElement('textarea');
      textarea.value = generatedPrompt;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleSave = async () => {
    setError('');
    setSaving(true);
    try {
      // Strip markdown code fences if present
      const cleaned = yamlInput
        .replace(/^```[\w]*\n?/, '')
        .replace(/\n?```\s*$/, '')
        .trim();

      const result = await saveDomainYaml(domainId, cleaned);
      if (result?.success) {
        onCreated();
      } else {
        setError(result?.error || result?.detail || t('admin.classification_save_error'));
      }
    } catch (e) {
      setError(e?.detail || e?.message || t('admin.classification_save_error'));
    }
    setSaving(false);
  };

  const stepTitles = [
    t('admin.classification_step1_title'),
    t('admin.classification_step2_title'),
    t('admin.classification_step3_title'),
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="bg-gray-800 border border-gray-600 rounded-xl w-full max-w-2xl max-h-[80vh] overflow-y-auto p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Step Indicator */}
        <div className="flex items-center gap-3 mb-6">
          {[1, 2, 3].map(s => (
            <div key={s} className="flex items-center gap-2">
              <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
                s === step ? 'bg-blue-600 text-white' : s < step ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-400'
              }`}>{s}</div>
              {s < 3 && <div className={`w-8 h-0.5 ${s < step ? 'bg-green-600' : 'bg-gray-700'}`} />}
            </div>
          ))}
          <span className="text-sm text-gray-300 ml-2">{stepTitles[step - 1]}</span>
        </div>

        {/* Step 1: Define Domain */}
        {step === 1 && (
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('admin.classification_domain_id')}</label>
              <input
                type="text"
                value={domainId}
                onChange={(e) => setDomainId(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, ''))}
                placeholder={t('admin.classification_domain_id_placeholder')}
                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              />
              {idError && <p className="text-xs text-red-400 mt-1">{idError}</p>}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">{t('admin.classification_domain_name_en')}</label>
                <input
                  type="text"
                  value={nameEn}
                  onChange={(e) => setNameEn(e.target.value)}
                  placeholder="e.g. Medical Image"
                  className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">{t('admin.classification_domain_name_ko')}</label>
                <input
                  type="text"
                  value={nameKo}
                  onChange={(e) => setNameKo(e.target.value)}
                  placeholder="예: 의료 이미지"
                  className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                />
              </div>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('admin.classification_domain_use_case')}</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder={t('admin.classification_domain_use_case_placeholder')}
                rows={4}
                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none resize-none"
              />
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <button onClick={onClose} className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors">
                {t('admin.classification_cancel')}
              </button>
              <button
                onClick={handleNext1}
                disabled={!canNext1}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed rounded text-sm text-white transition-colors"
              >
                {t('admin.classification_next')}
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Copy Prompt */}
        {step === 2 && (
          <div className="space-y-4">
            <p className="text-sm text-gray-300">{t('admin.classification_prompt_generated')}</p>
            <pre className="bg-gray-900 border border-gray-600 rounded p-3 text-xs text-gray-300 max-h-[40vh] overflow-y-auto whitespace-pre-wrap font-mono">
              {generatedPrompt}
            </pre>
            <div className="flex justify-between items-center pt-2">
              <button onClick={() => setStep(1)} className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors">
                {t('admin.classification_back')}
              </button>
              <div className="flex gap-2">
                <button
                  onClick={handleCopyPrompt}
                  className="flex items-center gap-1.5 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded text-sm text-white transition-colors"
                >
                  {copied ? <CheckCircle size={14} /> : <Copy size={14} />}
                  {copied ? t('admin.classification_prompt_copied') : t('admin.classification_copy_prompt')}
                </button>
                <button
                  onClick={() => setStep(3)}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm text-white transition-colors"
                >
                  {t('admin.classification_next')}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Paste YAML */}
        {step === 3 && (
          <div className="space-y-4">
            <label className="block text-xs text-gray-400">{t('admin.classification_paste_yaml')}</label>
            <textarea
              value={yamlInput}
              onChange={(e) => { setYamlInput(e.target.value); setError(''); }}
              rows={16}
              placeholder="domain:\n  id: ..."
              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-xs text-gray-200 placeholder-gray-600 focus:border-blue-500 focus:outline-none resize-none font-mono"
            />
            {error && (
              <div className="flex items-start gap-2 text-sm text-red-400 bg-red-900/20 border border-red-800/40 rounded p-3">
                <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}
            <div className="flex justify-between items-center pt-2">
              <button onClick={() => setStep(2)} className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors">
                {t('admin.classification_back')}
              </button>
              <button
                onClick={handleSave}
                disabled={saving || !yamlInput.trim()}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed rounded text-sm text-white transition-colors"
              >
                {saving ? t('admin.classification_saving') : t('admin.classification_save_domain')}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


// ── Classification Panel ─────────────────────────────────

export default function ClassificationPanel() {
  const { t } = useLocale();
  const [domains, setDomains] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [expandedTypes, setExpandedTypes] = useState(new Set());
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);

  const load = useCallback(async () => {
    try {
      const [domainList, activeConfig] = await Promise.all([
        listDomains(),
        getActiveDomainConfig(),
      ]);
      setDomains(domainList || []);
      const currentId = activeConfig?.active_domain || null;
      setActiveId(currentId);
      // Auto-select active domain or first
      const autoSelect = currentId || (domainList?.[0]?.id || null);
      if (autoSelect) {
        const detail = await getDomainDetail(autoSelect);
        setSelectedDomain(detail);
      }
    } catch (e) {
      console.error('Failed to load classification domains:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleActivate = async (domainId) => {
    if (saving || domainId === activeId) return;
    setSaving(true);
    try {
      await setActiveDomain(domainId);
      setActiveId(domainId);
    } catch (e) {
      console.error('Failed to activate domain:', e);
    }
    setSaving(false);
  };

  const handleSelectDomain = async (domainId) => {
    if (selectedDomain?.id === domainId) return;
    try {
      const detail = await getDomainDetail(domainId);
      setSelectedDomain(detail);
      setExpandedTypes(new Set());
    } catch (e) {
      console.error('Failed to load domain detail:', e);
    }
  };

  const toggleType = (type) => {
    setExpandedTypes(prev => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  if (loading) return <div className="text-gray-400 text-sm">{t('status.loading')}</div>;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold">{t('admin.classification_title')}</h2>
          <p className="text-sm text-gray-400 mt-1">{t('admin.classification_desc')}</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm text-white transition-colors flex-shrink-0"
        >
          <Plus size={14} />
          {t('admin.classification_create')}
        </button>
      </div>

      {/* Domain Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-6">
        {domains.map(d => {
          const isActive = d.id === activeId;
          const isSelected = d.id === selectedDomain?.id;
          return (
            <div
              key={d.id}
              onClick={() => handleSelectDomain(d.id)}
              className={`bg-gray-800 rounded-lg border p-4 cursor-pointer transition-colors ${
                isSelected
                  ? isActive ? 'border-green-500 ring-1 ring-green-500/30' : 'border-blue-500 ring-1 ring-blue-500/30'
                  : isActive ? 'border-green-500/50' : 'border-gray-700 hover:border-gray-500'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-medium truncate">{d.name}</h3>
                {isActive ? (
                  <span className="bg-green-900/50 text-green-300 rounded px-2 py-0.5 text-xs flex-shrink-0">
                    {t('admin.classification_active')}
                  </span>
                ) : (
                  <button
                    onClick={(e) => { e.stopPropagation(); handleActivate(d.id); }}
                    disabled={saving}
                    className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded text-xs px-3 py-1 flex-shrink-0 transition-colors"
                  >
                    {saving ? '...' : t('admin.classification_activate')}
                  </button>
                )}
              </div>
              <p className="text-xs text-gray-400 mb-2 line-clamp-2">{d.description}</p>
              <div className="text-xs text-gray-500">
                {t('admin.classification_types_count', { count: d.image_types_count || d.image_types?.length || 0 })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected Domain Detail */}
      {selectedDomain ? (
        <div className="space-y-4">
          {/* Image Types */}
          <div>
            <h3 className="text-sm font-medium text-gray-300 mb-2">{t('admin.classification_image_types')}</h3>
            <div className="flex flex-wrap gap-2">
              {(selectedDomain.image_types || []).map(type => (
                <span
                  key={type}
                  className="bg-blue-900/50 text-blue-300 rounded px-2.5 py-1 text-xs"
                >
                  {type}
                </span>
              ))}
            </div>
          </div>

          {/* Type Hints Accordion */}
          <div>
            <h3 className="text-sm font-medium text-gray-300 mb-2">{t('admin.classification_type_hints')}</h3>
            <div className="bg-gray-800 rounded-lg border border-gray-700 divide-y divide-gray-700">
              {Object.entries(selectedDomain.type_hints || {}).map(([type, hints]) => (
                <div key={type}>
                  <button
                    onClick={() => toggleType(type)}
                    className="w-full flex items-center gap-2 px-4 py-3 text-sm hover:bg-gray-700/30 transition-colors"
                  >
                    <ChevronDown
                      size={14}
                      className={`text-gray-400 transition-transform ${expandedTypes.has(type) ? '' : '-rotate-90'}`}
                    />
                    <span className="font-medium text-gray-200">{type}</span>
                    <span className="text-xs text-gray-500 ml-auto">
                      {Object.keys(hints).length} hints
                    </span>
                  </button>
                  {expandedTypes.has(type) && (
                    <div className="px-4 pb-3 space-y-1.5">
                      {Object.entries(hints).length > 0 ? (
                        Object.entries(hints).map(([key, values]) => (
                          <div key={key} className="flex gap-2 text-xs">
                            <span className="text-gray-400 min-w-[120px] flex-shrink-0">{key}:</span>
                            <span className="text-gray-300">
                              {Array.isArray(values) ? values.join(', ') : String(values)}
                            </span>
                          </div>
                        ))
                      ) : (
                        <div className="text-xs text-gray-500">{t('admin.classification_no_hints')}</div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Common Hints */}
          {selectedDomain.common_hints && Object.keys(selectedDomain.common_hints).length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-300 mb-2">{t('admin.classification_common_hints')}</h3>
              <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-1.5">
                {Object.entries(selectedDomain.common_hints).map(([key, values]) => (
                  <div key={key} className="flex gap-2 text-xs">
                    <span className="text-gray-400 min-w-[120px] flex-shrink-0">{key}:</span>
                    <span className="text-gray-300">
                      {Array.isArray(values) ? values.join(', ') : String(values)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Type Instructions */}
          {selectedDomain.type_instructions && Object.keys(selectedDomain.type_instructions).length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-300 mb-2">{t('admin.classification_type_instructions')}</h3>
              <div className="bg-gray-800 rounded-lg border border-gray-700 divide-y divide-gray-700">
                {Object.entries(selectedDomain.type_instructions).map(([type, instruction]) => (
                  <div key={type} className="px-4 py-3">
                    <span className="text-xs font-medium text-gray-200">{type}</span>
                    <p className="text-xs text-gray-400 mt-1">{instruction}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-sm text-gray-500">{t('admin.classification_no_domain')}</div>
      )}

      {/* Create Domain Modal */}
      {showCreateModal && (
        <CreateDomainModal
          existingIds={domains.map(d => d.id)}
          onClose={() => setShowCreateModal(false)}
          onCreated={() => { setShowCreateModal(false); load(); }}
        />
      )}
    </div>
  );
}
