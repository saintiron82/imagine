/**
 * 분류/도메인 (IMGV2-17) — 분석 분류 기준(도메인) 조회·활성화·생성.
 * bridge.js 의 도메인 함수(local Electron / remote HTTP 분기 내장)를 TanStack Query 로 감싼다.
 *   listDomains()            GET  /api/v1/admin/classification/domains
 *   getDomainDetail(id)      GET  /api/v1/admin/classification/domains/{id}
 *   getActiveDomainConfig()  GET  /api/v1/admin/classification/active
 *   setActiveDomain(id)      PUT  /api/v1/admin/classification/active
 *   saveDomainYaml(id, yaml) POST /api/v1/admin/classification/domains  (생성 전용 — 기존 id 는 409)
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  listDomains, getDomainDetail, getActiveDomainConfig, setActiveDomain, saveDomainYaml,
} from '../services/bridge'

export function useDomains() {
  return useQuery({ queryKey: ['domains'], queryFn: listDomains })
}

export function useActiveDomain() {
  return useQuery({ queryKey: ['domain-active'], queryFn: getActiveDomainConfig })
}

export function useDomainDetail(domainId) {
  return useQuery({
    queryKey: ['domain', domainId],
    queryFn: () => getDomainDetail(domainId),
    enabled: !!domainId,
  })
}

export function useSetActiveDomain() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (domainId) => setActiveDomain(domainId),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['domain-active'] }) },
  })
}

export function useSaveDomain() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ domainId, yamlContent }) => saveDomainYaml(domainId, yamlContent),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['domains'] }) },
  })
}

/**
 * 새 도메인 생성용 AI 프롬프트 생성 (구 ClassificationPanel 의 generateDomainPrompt 이식).
 * 운영자가 이 프롬프트를 LLM 에 넣어 YAML 을 받아오면 그대로 붙여넣어 저장한다.
 */
export function generateDomainPrompt({ domainId, nameEn, nameKo, description }) {
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

## Rules
1. Output ONLY the YAML content — no markdown fences, no explanations, no commentary
2. image_types: 3-12 items, lowercase snake_case
3. type_hints: every image_type MUST have at least 1 hint field with 3-15 values
4. Hint field names and values: snake_case
5. type_instructions: provide for at least the top 3 most complex types
6. Do NOT include common_hints (art_style, color_palette) — those come from _base.yaml automatically
7. domain.id MUST be exactly: ${domainId}
8. Make the hints genuinely useful for classifying "${nameEn}" images`
}
